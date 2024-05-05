import os
import random
import uuid
from pathlib import Path
from typing import Callable, Any, Optional, Dict

from TTS.tts.models.xtts import Xtts
from llama_cpp import ChatCompletionRequestMessage, Llama
from mypy_boto3_s3.client import S3Client

from .reply import Reply
from .request import Request
from .stage import Stage
from ..lipsync import Wav2Lip
from ..llm import create_chat_completion
from ..personalities import load_personality
from ..tts.xtts import inference as xtts
from ..utils import strict_getenv, flatten
from .types import ResponseType

bucket_videos = strict_getenv('S3_BUCKET_VIDEOS')


def upload_to_s3(s3: S3Client, path: Path, bucket_name: str, key: str) -> str:
    s3.upload_file(str(path), bucket_name, key)
    os.remove(path)
    return s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket_videos, 'Key': key},
        ExpiresIn=7200
    )


def create_handler(
        s3: S3Client,
        llm: Llama,
        xtts_model: Xtts,
        wav2lip: Wav2Lip
) -> Callable[[dict[str, Any], Stage], dict[str, Any]]:
    def handler(job: dict[str, Any], stage: Stage) -> dict[str, Any]:
        stage.update_stage('Получение данных личности из хранилища')
        request = Request.model_validate(job['input'])
        personality = load_personality(s3, xtts_model, request.personality)

        messages: list[ChatCompletionRequestMessage] = [
                                                           {"role": "system", "content": personality.prompt},
                                                       ] + flatten([
            [
                {"role": "user", "content": message.request},
                {"role": "assistant", "content": message.reply}
            ]
            for message in request.messages
        ]) + [
                                                           {"role": "user", "content": request.prompt},
                                                       ]

        stage.update_stage('Создание текстового ответа')
        result = create_chat_completion(llm, messages)
        text: Optional[str] = result["choices"][0]["message"]["content"]

        if text is None:
            raise RuntimeError("result from llm is null")

        if request.response_type == ResponseType.TEXT:
            return Reply(
                text=text,
                object_type=request.response_type,
                object=None,
                object_url=None
            ).model_dump(mode='json')

        stage.update_stage('Создание аудио ответа')
        generated_audio = Path(f"/tmp/{uuid.uuid4()}.wav")
        xtts(
            xtts_model,
            text,
            personality.gpt_cond_latent,
            personality.speaker_embedding,
            generated_audio,
            language=str(personality.language)
        )

        if request.response_type == ResponseType.AUDIO:
            object_id = str(uuid.uuid4()) + ".wav"
            return Reply(
                text=text,
                object_type=request.response_type,
                object=object_id,
                object_url=upload_to_s3(s3, generated_audio, bucket_videos, object_id)
            ).model_dump(mode='json')

        stage.update_stage('Процесс синхронизации губ на видео')
        video = random.choice(personality.video_objects)
        generated_video = Path(f"/tmp/{uuid.uuid4()}.mp4")

        wav2lip.process(
            str(generated_audio),
            str(video),
            str(generated_video),
            personality.enhance,
            personality.female,
            stage
        )

        stage.update_stage('Загрузка файла в хранилище')

        os.remove(generated_audio)
        object_id = str(uuid.uuid4()) + ".mp4"
        url = upload_to_s3(s3, generated_video, bucket_videos, object_id)

        stage.update_stage('Генерация завершена', object_url=url)

        return Reply(
            text=text,
            object_type=request.response_type,
            object=object_id,
            object_url=url
        ).model_dump(mode='json')

    return handler
