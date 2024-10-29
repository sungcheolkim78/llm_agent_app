# Sung-Cheol Kim Copyright 2024, All rights reserved.

import instructor
from pydantic import BaseModel, Field
from openai import OpenAI
from anthropic import Anthropic
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import wraps
from time import time


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


# Define your desired output structure
class UserInfo(BaseModel):
    name: str
    age: int
    fact: List[str] = Field(..., description="A list of facts about the character")


# Define hook functions
def log_kwargs(**kwargs):
    print(f"Function called with kwargs: {kwargs}")


def log_exception(exception: Exception):
    print(f"An exception occurred: {str(exception)}")


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time()
        result = f(*args, **kwargs)
        end_time = time()
        print(f"Function {f.__name__} took {end_time - start_time:2.4f} seconds to execute")
        return result

    return wrapper


@timing
def get_from_openai(settings):
    client = instructor.from_openai(OpenAI(api_key=settings.OPENAI_API_KEY))
    user_info = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserInfo,
        messages=[{"role": "user", "content": "John Doe is 30 years old."}],
    )
    return user_info


@timing
def get_from_anthropic(settings):
    client = instructor.from_anthropic(Anthropic(api_key=settings.ANTHROPIC_API_KEY))
    resp = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Extract Jason is 25 years old."}],
        response_model=UserInfo,
    )
    return resp


@timing
def get_from_ollama(settings, hook_on: bool = False):
    client = instructor.from_openai(
        OpenAI(base_url="http://192.168.1.4:11434/v1", api_key="ollama"),
        mode=instructor.Mode.JSON,
    )

    if hook_on:
        client.on("completion:kwargs", log_kwargs)
        client.on("completion:error", log_exception)

    resp = client.chat.completions.create(
        model="aya-expanse:8b",
        messages=[{"role": "user", "content": "Tell me about the Harry Potter"}],
        response_model=UserInfo,
    )
    return resp


def main():
    settings = Settings()

    calls = [get_from_openai, get_from_anthropic, get_from_ollama]

    for call in calls:
        user_info = call(settings)
        print(user_info.name)
        print(user_info.age)


if __name__ == "__main__":
    main()
