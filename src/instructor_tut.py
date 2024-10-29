# Sung-Cheol Kim Copyright 2024, All rights reserved.

import instructor
from pydantic import BaseModel, Field
from openai import OpenAI
from anthropic import Anthropic
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')


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


def get_from_openai(settings):
    client = instructor.from_openai(OpenAI(api_key=settings.OPENAI_API_KEY))
    user_info = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserInfo,
        messages=[{"role": "user", "content": "John Doe is 30 years old."}],
    )
    return user_info


def get_from_anthropic(settings):
    client = instructor.from_anthropic(Anthropic(api_key=settings.ANTROPIC_API_KEY))
    resp = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Extract Jason is 25 years old."}],
        response_model=UserInfo,
    )
    return resp


def get_from_ollama(settings, hook_on:bool = False):
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


settings = Settings()
user_info = get_from_ollama(settings)
print(user_info.name)
print(user_info.age)
