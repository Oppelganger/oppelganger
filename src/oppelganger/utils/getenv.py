import os


def strict_getenv(name: str) -> str:
	value: str | None = os.getenv(name)
	if value is None:
		raise RuntimeError(f'Environment variable {name} is not set')
	return value
