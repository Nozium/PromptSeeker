## Use Open API
import openai

import PromptSeeker.modules.config as CONFIG

API_KEY = CONFIG.COMMON_TOKEN
MODERATE_CATEGORY_SCORE = CONFIG.MODERATE_CATEGORY_SCORE


class OpenAIWrapper(object):
    def __init__(self, api_key=API_KEY, engine="gpt-3.5-turbo", max_retry=3) -> None:
        super().__init__()
        self.openai = openai
        self.openai.api_key = api_key
        self.engine = engine
        self.max_retry = max_retry

    def _davinchi(
        self,
        prompt,
        max_tokens=100,
        temperature=0.5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n", " Human:", " AI:"],
    ):
        return self.openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
        )

    def _ChatGpt(
        self,
        prompt,
        max_tokens=100,
        temperature=0.5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n", " Human:", " AI:"],
    ):
        """GPT general call function"""
        # print(prompt)
        time.sleep(1)
        return self.openai.ChatCompletion.create(
            model=self.engine,
            messages=prompt,  # this may need to contain a list of messages
        )

    def moderate(self, res_text):
        """Moderate the prompt"""
        return self.openai.Moderation.create(input=res_text)

    def ask(self, use_common_moderation=False, **kwargs):
        retries = 0
        while retries < self.max_retry:
            try:
                if self.engine == "davinci":
                    res = self._davinchi(**kwargs)
                elif "gpt" in self.engine:
                    res = self._ChatGpt(**kwargs)
                else:
                    raise ValueError("Engine not supported")

                if res.choices[0].message.content:
                    content = res.choices[0].message.content
                else:
                    content = res.choices[0].text

                print(content)
                if use_common_moderation:
                    moderate_score = self.moderate(content)
                    c_score = moderate_score["results"][0]["category_scores"]
                    for k, v in c_score.items():
                        if v > MODERATE_CATEGORY_SCORE.get(k, 1):
                            raise ValueError("Moderation failed at {}".format(k))
                return content
            except openai.error.APIConnectionError as e:
                print(
                    f"APIConnectionError: {e}. Retrying... ({retries + 1}/{self.max_retry})"
                )
                retries += 1
                if retries < self.max_retry:
                    time.sleep(2**retries)  # Exponential backoff
                else:
                    raise e  # すべてのリトライが失敗した場合、エラーを再度送出します

    def validate_output(self, prompt_structure, llm_response):
        """"""
        return bool, None
