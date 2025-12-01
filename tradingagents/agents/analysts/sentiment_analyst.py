from datetime import datetime, timedelta

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import get_news


def create_sentiment_analyst(llm):
    def sentiment_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        start_date = (
            datetime.fromisoformat(current_date) - timedelta(days=7)
        ).date()

        tools = [
            get_news,
        ]

        system_message = (
            "You are a sentiment analyst focused on gauging public mood and narrative momentum around a specific company over the past week."
            " Always start by calling get_news with ticker={ticker}, start_date={start_date}, end_date={current_date} to fetch the latest discussions before forming conclusions."
            " Emphasize sentiment drivers, momentum shifts, and investor perception. Avoid generic statements; deliver specific observations that can help traders anticipate sentiment-driven moves."
            " Append a concise Markdown table summarizing key sentiment drivers and their potential impact."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. The current company we want to analyze is {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)
        prompt = prompt.partial(start_date=str(start_date))

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "sentiment_report": report,
        }

    return sentiment_analyst_node
