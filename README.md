# Personality Engine

Environment variables:
- S3_ENDPOINT_URL
- S3_BUCKET
- S3_KEY_ID
- S3_ACCESS_KEY

Building:
```shell
$ docker buildx build . --target final --build-arg CUDA_VERSION=12.2.2
```
