from typing import List

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openrouter import ChatOpenRouter
from phoenix.otel import register
from pydantic import BaseModel, Field

from langchain.agents import create_agent

register(project_name="cat_color_and_age", auto_instrument=True)
load_dotenv()


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="Thr agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


def main() -> None:
    prompt_template = PromptTemplate(template="Say one fast pokemon curiosity")

    llm = ChatOpenRouter(temperature=0.7, model="deepseek/deepseek-v4.1-flash")
    agent = create_agent(model=llm, response_format=AgentResponse)
    structured_model = llm.with_structured_output(AgentResponse, method="json_schema")

    response = structured_model.invoke(prompt_template.format())

    print(response)


main()
