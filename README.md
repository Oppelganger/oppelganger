# Personality Engine

Environment variables:
- S3_ENDPOINT_URL
- S3_BUCKET_VIDEOS
- S3_BUCKET_PERSONALITIES
- S3_REGION_NAME
- S3_ACCESS_KEY_ID
- S3_SECRET_ACCESS_KEY
- S3_SESSION_TOKEN

Building:
```shell
$ docker buildx build . --target final --build-arg CUDA_VERSION=12.2.2
```
Do not forget `-t tag`
