# -*- coding: utf-8 -*-
import json
import os
import re
import sys
import time

sys.path.append(os.getcwd())

from PromptSeeker.modules.openaiwappper import OpenAIWrapper


class PromptSeek(object):
    def __init__(
        self,
        goal,
        llm_wrapper: OpenAIWrapper,
        max_process=10,
    ) -> None:
        self.LLM = llm_wrapper
        self.goal = goal
        self.max_process = max_process
        self.goal_contents = []
        self.decomposed_steps = []
        self.variables = []
        self.variables_description = {}
        self.step_prompts = []
        self.process_count = 0
        self.plane_optimization = ""
        self.plane_decomposition = ""
        self.plane_redefinition = ""
        self.plane_step_prompts = ""

    def to_dict(self):
        return {
            "goal": self.goal,
            "goal_contents": self.goal_contents,
            "decomposed_steps": self.decomposed_steps,
            "plane_decomposed_steps": self.plane_decomposition,
            "variables": self.variables,
            "plane_optimization": self.plane_optimization,
            "variables_description": self.variables_description,
            "plane_redefinition": self.plane_redefinition,
            "step_prompts": self.step_prompts,
            "plane_step_prompts": self.plane_step_prompts,
            "process_count": self.process_count,
        }

    def to_formated_text(self):
        formated_text = ""
        for k, v in self.to_dict().items():
            if "plane" in k:
                continue
            if type(v) == list:
                flatten_v = "\n".join(v)
            else:
                flatten_v = v
            formated_text += f"{k}:\n{flatten_v}\n\n"
        return formated_text

    def save(
        self,
        save_dir="./results/prompt_seeks/",
        save_name="prompt_seek.json",
        with_goal=True,
    ):
        os.makedirs(save_dir, exist_ok=True)
        now = time.strftime("%Y%m%d%H%M%S", time.localtime())
        if with_goal:
            save_path = save_dir + now + self.goal.strip() + save_name
        else:
            save_path = save_dir + now + save_name
        with open(save_path, "w") as f:
            json.dump(self.to_dict(), f)

    def seek(self):
        self.decompose_goal()
        self.optimize_variables()
        self.redefine_goal_and_variables()
        self.generate_step_prompts()
        self.save()
        return self.get_final_prompt()

    def auto_seek(self, max_process=None):
        if max_process is None:
            max_process = self.max_process
        self.decompose_goal()
        self.optimize_variables()
        self.redefine_goal_and_variables()
        self.generate_step_prompts()
        self.save()
        # base case: reached maximum number of processes
        if max_process is not None and self.process_count >= max_process:
            return self.get_final_prompt()

        # recursive case: continue calling function
        self.process_count += 1
        return self.auto_seek(max_process)

    def get_prompt_seek_rules(self):
        return self.LLM.ask(prompt="What are the rules of the prompt seek method?")

    def decompose_goal(self):
        """Decompose the goal into a list of prompts
        - [コンテンツの詳細]:
            - [Goal] : 小説を生成するためのChatGPTを使ったシステムのOutlineを作成すること。
            - Goalを運成するために必要な手順を分解します。
            - 分解した手順は「Pn」に順番にNenberを付けて格納していきます。
            - 変数を定義します。
        - [C1] :
            - Goalを達成するために必量なことをStep by Stepで1つづつ実行していけるように手順:[P#]に分解して下さい。
            - [Output style] :
                - [P1] =
                ...
                - [P#] =
                ...
                [P{END}] =
        """

        decomposition_rule = ("").join(
            [
                "You are ChatGPT-5 simulator, and well performed coach and assistant.\n",
                "Now you coach and assist the user to decompose the goal into step-by-step prompts.\n",
                "Through this process, you will coach how to decompose the goal and realize it.\n",
            ]
        )

        first_dummy_prompt = (
            f"I want to decompose the goal {self.goal} into a list of prompts"
        )
        if self.process_count == 0:
            dummy_user_prompt = first_dummy_prompt
            previous_decomposition = ("").join(
                [
                    f"Decompose the goal into a list of prompts: {self.goal} into bellow format\n",
                    "- Deteil of the goal contents\n",
                    "   Please break it down into steps: [P#] so that you can do what is necessary to achieve the goal step by step.\n",
                    "   - [Output style] : python dict case as bellow\n",
                    "       P1 : value\n",
                    "       ...\n",
                    "       P# : value\n",
                    "       ...\n",
                    "       PEND : value\n",
                ]
            )
        else:
            # TODO : add previous decomposition
            dummy_user_prompt = f"Please brushup list of prompts for the goal {self.goal} in same format"

            _prev = [f"[P{i}]:" + d + "\n" for i, d in enumerate(self.decomposed_steps)]

            previous_decomposition = ("").join(
                [
                    f"Decompose the goal into a list of prompts: {self.goal} into bellow format\n",
                    "- Deteil of the goal contents\n",
                    "   Please break it down into steps: [P#] so that you can do what is necessary to achieve the goal step by step.\n",
                    "   - [Output style] : python dict case as bellow\n",
                ]
                + _prev
            )

        decomposition = self.LLM.ask(
            prompt=[
                {"role": "system", "content": decomposition_rule},
                {"role": "user", "content": first_dummy_prompt},
                {"role": "assistant", "content": previous_decomposition},
                {"role": "user", "content": dummy_user_prompt},
            ]
        )
        steps, variables = self._parse_decomposition(decomposition)
        self.decomposed_steps = steps
        self.variables = variables

    def optimize_variables(self):
        """Optimize the variables
        - [C2] :
            - 各種変数を使用して、変数を減らすことができないか検対する
            - [Gaol]は必要条件として必ずInputする。
            - [Gaol]の定義を変数を使用して表すことで、[Gaol]の定義だけで手順を分解できるようにしたい
            - 一般化して、変数を追加して[Goal]の定義を書き表して下さい
            - [Output style] :
                - [Added variable] をリスト形式で一般化して書き出して下さい
                - 続けて、[Goal]の定義を（[Added variable]を使用して書き出して下さい
                - [Gaol]:{Gaol}
                - 追加の変数を質問して下さい。一つずつ定義を書き表して下さい
        """
        optimization_rule = "".join(
            [
                "Now you coach and assist the user to optimize the variables and redefine the goal.\n",
                "Through this process, you will coach how to optimize the variables and redefine the goal.\n",
            ]
        )

        first_dummy_prompt = f"Please optimize the variables and provide guidance for redefining the goal {self.goal} ."
        if self.process_count == 0:
            dummy_user_prompt = first_dummy_prompt
            previous_decomposition = ("").join(
                [
                    f"[Optimize the variables]: {', '.join(self.variables)} into bellow format",
                    "- [Goal] must be inputted as a necessary condition\n",
                    "- We want to decompose the procedure based solely on the definition of [Goal] by expressing the definition of [Goal] using variables.\n",
                    "- Generalize and add variables to express the definition of Goal.\n",
                    "   - [Output style] : json cases,\n",
                    "       - Generalize dict as [Added variable] key and [Added variable] value dict.\n",
                    "       - Next, write the definition of Goal using [Added variable]. \n",
                    "       - Ask for additional variables and write their definitions one by one.\n",
                ]
            )
        else:
            dummy_user_prompt = first_dummy_prompt
            previous_decomposition = ("").join(
                [
                    f"[Optimize the variables]: {', '.join(self.variables)} into bellow format",
                    "- [Goal] must be inputted as a necessary condition\n",
                    "- We want to decompose the procedure based solely on the definition of [Goal] by expressing the definition of [Goal] using variables.\n",
                    "- Generalize and add variables to express the definition of Goal.\n",
                    "   - [Output style] : json cases,\n",
                    "       - Generalize dict as [Added variable] key and [Added variable] value dict.\n",
                    "       - Next, write the definition of Goal using [Added variable]. \n",
                    "       - Ask for additional variables and write their definitions one by one.\n",
                ]
            )

        optimization = self.LLM.ask(
            prompt=[
                {"role": "system", "content": optimization_rule},
                {"role": "user", "content": first_dummy_prompt},
                {"role": "assistant", "content": previous_decomposition},
                {
                    "role": "user",
                    "content": dummy_user_prompt,
                },
            ]
        )
        self.variables_description = self._parse_optimization(optimization)

    def redefine_goal_and_variables(self):
        redefinition_rule = "".join(
            [
                "Now you coach and assist the user to redefine the goal and update the variables.\n",
                "Through this process, you will coach how to redefine the goal and update the variables.\n",
            ]
        )
        first_dummy_prompt = ("").join(
            [
                f"[Redefine the goal and variables] : {self.goal}, {', '.join(self.variables)}  into bellow format.\n"
                "   - Interpret the variables defined in [Optimize the variables] generally and supplement them.\n",
                "   - Redefine [Goal] using the supplemented variables.\n",
                "   - [Output style] :\n",
                f"       - Write the redefined [Goal] using [Goal]: [new_goal_description].\n",
            ]
        )
        prev_ = [self.goal] + [
            f"{k}:{v}" for k, v in self.variables_description.items()
        ]

        previous_results = ("").join(prev_) if len(prev_) > 0 else ""

        dummy_user_prompt = "Please help me redefine the goal and update the variables."
        redefinition = self.LLM.ask(
            prompt=[
                {"role": "system", "content": redefinition_rule},
                {"role": "user", "content": first_dummy_prompt},
                {"role": "assistant", "content": previous_results},
                {
                    "role": "user",
                    "content": dummy_user_prompt,
                },
            ]
        )
        self.goal_contents, self.variables = self._parse_redefinition(redefinition)

    def generate_step_prompts(self):
        """Generate the step prompt
        - [C4] :
            - [コンテンツの詳細]を元に[Gaol]を運成するために、Step by Stepで実行していきます。
                - [P1]から[P#]を経て順番に[P{END}]まで実行していきます。
                - [Output style] :
                    - [O1] = {Output[P1]}
                    ...
                    - [O#] = {Output[P#]}
                    ...
                    - [O{END}] = {Output[P{END}]}
        """
        generation_rule = "".join(
            [
                "Now you coach and assist the user to generate step prompts based on the decomposed steps.\n",
                "Through this process, you will coach how to generate step prompts based on the decomposed steps.\n",
            ]
        )

        first_dummy_prompt = ("").join(
            [
                "- [generate_step_prompts] :\n",
                "   - To achieve [Goal] based on [Details of Content], execute step by step.\n",
                "       - Execute from [P1] to [PEND] in order through [P1] to [P#].\n",
                "       - [Output style] : python dict \n",
                "           - [O1] : Output of [P1]\n",
                "           ...\n",
                "           - [O#] = Output of [P#]}\n",
                "           ...\n",
                f"           - [OEND] = Output of [PEND]\n",
            ]
        )
        if self.decomposed_steps:
            previous_results = ("").join(
                [
                    f"[O{i}] = Output of {step}\n"
                    for i, step in enumerate(self.decomposed_steps)
                ]
            )
        else:
            previous_results = "No decomposed steps"

        for step_id, step in enumerate(self.decomposed_steps):
            step_id = step_id if step_id < len(self.decomposed_steps) else "END"
            step_prompt = self.LLM.ask(
                prompt=[
                    {"role": "system", "content": generation_rule},
                    {"role": "user", "content": first_dummy_prompt},
                    {"role": "assistant", "content": previous_results},
                    {
                        "role": "user",
                        "content": f"Please help me generate the step prompt for step {step_id}: {step}",
                    },
                ]
            )
            self.step_prompts.append(step_prompt)

    ### 移植前
    def get_final_prompt(self):
        return ("\n").join(self.step_prompts)

    # def _parse_decomposition(self, decomposition: str):
    #     self.plane_decomposition = decomposition
    #     try:
    #         data = json.loads(decomposition)
    #         steps = data.get("steps", [])
    #         variables = data.get("variables", [])
    #     except:
    #         # if decompotion is not json format, it is a string
    #         lines = decomposition.strip().split("\n")
    #         steps = []
    #         variables = []
    #         for line in lines:
    #             if line.startswith("[P"):
    #                 steps.append(line)
    #             elif line.startswith("[V"):
    #                 variables.append(line)
    #     return steps, variables

    def _parse_decomposition(self, decomposition: str):
        self.plane_decomposition = decomposition

        # case 1 :json format
        try:
            data = json.loads(decomposition)
            steps = data.get("steps", [])
            variables = data.get("variables", [])
        # case 2 : string format (regular expression)
        except:
            # Regular expressions to match steps and their descriptions in the decomposition
            step_pattern = re.compile(r" P(?:\d+|END): (.*?)\n", re.MULTILINE)
            description_pattern = re.compile(
                r" P(?:\d+|END): .*?\n(.*?)(?=\n P(?:\d+|END)|$)",
                re.MULTILINE | re.DOTALL,
            )

            steps = step_pattern.findall(decomposition)
            descriptions = [
                desc.strip() for desc in description_pattern.findall(decomposition)
            ]

            variables = dict(zip(steps, descriptions))

        # case 3 : not json format, not string format
        if not steps or not variables:
            lines = decomposition.strip().split("\n")
            steps = []
            variables = []
            mode = ""
            for l in lines:
                # decomposition returns
                # - deocomposition step summary
                #    - [Output] : [P#] : description
                if "- " in l and ":" in l:
                    _l = l.split(":")
                    step, description = _l[-2], _l[-1]
                    if mode != "-to[":
                        steps.append(step.strip())
                    variables.append(description.strip())
                elif "- " in l:
                    if not steps:
                        mode = "-to["

                    if mode == "-to[":
                        steps.append(l.replace("- ", "").strip())
                    else:
                        variables.append(l.replace("- ", "").strip())

        # case4 : not json format, not string format, not regular expression
        if not steps or not variables:
            steps = [l for l in decomposition.strip().split("\n")[1]]
            variables = [l for l in decomposition.strip().split("\n")[2:]]
        return steps, variables

    def _parse_optimization(self, optimization: str):
        self.plane_optimization = optimization
        try:
            data = json.loads(optimization)
            variables_description = data.get("variables_description", {})
        except:
            lines = optimization.strip().split("\n")
            variables_description = {}
            for line in lines:
                if line.startswith("[V"):
                    var_name, description = line.split(":", 1)
                    variables_description[var_name.strip()] = description.strip()
        return variables_description

    def _parse_redefinition(self, redefinition: str):
        self.plane_redefinition = redefinition
        try:
            data = json.loads(redefinition)
            goal_contents = data.get("goal_contents", "")
            variables = data.get("variables", [])
        except:
            lines = redefinition.strip().split("\n")
            goal_contents = []
            variables = []
            for line in lines:
                if line.startswith("[Goal"):
                    goal_contents.append(line)
                elif line.startswith("[V"):
                    variables.append(line)
        return goal_contents, variables


if __name__ == "__main__":
    goal = "To build FastAPI application pytest generator."
    open_ai_wapper = OpenAIWrapper()
    prompt_seeker = PromptSeek(
        goal=goal,
        llm_wrapper=open_ai_wapper,
        max_process=10,
    )
    first_prompt = prompt_seeker.seek()
    print(first_prompt)
    final_step_prompt = prompt_seeker.auto_seek()
    print(final_step_prompt)
    print("-------------final prompt--------------")
    formated_text = prompt_seeker.to_formated_text()
    print(formated_text)
