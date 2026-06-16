"""
AI多维度出题系统 - Prompt构造器（方案B核心 - 销售培训版）
基于LLM Prompt的智能出题引擎，支持同一知识点多角度变体生成
"""

import random
import hashlib
from typing import List, Optional

from models import (
    KnowledgeLevel,
    QuestionDimension,
    QuestionType,
    GenerationContext,
)


class PromptBuilder:
    """
    多维度Prompt构造器 - 方案B的核心组件（销售培训版）
    
    设计理念：
    1. 系统Prompt定义销售培训出题原则和质量标准
    2. 用户Prompt根据上下文动态构造，包含差异化指令
    3. 变体指令确保每个学员拿到不同的题目（防抄袭）
    """

    # 维度到难度的映射：不同提问角度适合不同认知深度
    DIMENSION_DIFFICULTY_MAP = {
        QuestionDimension.CONCEPT: [KnowledgeLevel.REMEMBER, KnowledgeLevel.UNDERSTAND],
        QuestionDimension.CASE_ANALYSIS: [KnowledgeLevel.UNDERSTAND, KnowledgeLevel.APPLY],
        QuestionDimension.ROLE_PLAY: [KnowledgeLevel.UNDERSTAND, KnowledgeLevel.APPLY],
        QuestionDimension.OBJECTION_HANDLING: [KnowledgeLevel.APPLY, KnowledgeLevel.ANALYZE],
        QuestionDimension.TECHNIQUE_COMPARISON: [KnowledgeLevel.ANALYZE, KnowledgeLevel.EVALUATE],
        QuestionDimension.SCENARIO_CHOICE: [KnowledgeLevel.APPLY, KnowledgeLevel.ANALYZE, KnowledgeLevel.EVALUATE],
        QuestionDimension.PRACTICE_APPLICATION: [KnowledgeLevel.APPLY, KnowledgeLevel.ANALYZE],
    }

    # 维度到题型的映射
    DIMENSION_TYPE_MAP = {
        QuestionDimension.CONCEPT: [
            QuestionType.SINGLE_CHOICE,
            QuestionType.MULTIPLE_CHOICE,
            QuestionType.JUDGE,
            QuestionType.SHORT_ANSWER,
        ],
        QuestionDimension.CASE_ANALYSIS: [
            QuestionType.SINGLE_CHOICE,
            QuestionType.SHORT_ANSWER,
            QuestionType.CASE_FILL,
        ],
        QuestionDimension.ROLE_PLAY: [
            QuestionType.DIALOGUE_FIX,
            QuestionType.SHORT_ANSWER,
            QuestionType.FILL_BLANK,
        ],
        QuestionDimension.OBJECTION_HANDLING: [
            QuestionType.SINGLE_CHOICE,
            QuestionType.SHORT_ANSWER,
            QuestionType.CASE_FILL,
        ],
        QuestionDimension.TECHNIQUE_COMPARISON: [
            QuestionType.SINGLE_CHOICE,
            QuestionType.MULTIPLE_CHOICE,
            QuestionType.SHORT_ANSWER,
        ],
        QuestionDimension.SCENARIO_CHOICE: [
            QuestionType.SINGLE_CHOICE,
            QuestionType.SHORT_ANSWER,
        ],
        QuestionDimension.PRACTICE_APPLICATION: [
            QuestionType.SINGLE_CHOICE,
            QuestionType.SHORT_ANSWER,
            QuestionType.CASE_FILL,
        ],
    }

    # 变体策略池 - 随机抽取2种来差异化每个学员的题目
    VARIATION_STRATEGIES = [
        "请使用不同的客户场景/行业背景，避免与其他学员雷同",
        "请使用不同的销售案例和数据示例",
        "请从不同的角度切入该销售知识点",
        "请设计不同的客户对话场景，展示该知识点的不同侧面",
        "请设计含有常见错误话术/误区的干扰项，基于真实的销售失误案例",
        "请将知识点放在不同的业务场景中考查（如B2B/B2C/大客户/零售等）",
        "请改变题干的表述方式，用不同的句式结构",
        "请调整选项的数量和排列顺序",
    ]

    @staticmethod
    def build_system_prompt() -> str:
        """构建系统级Prompt - 定义AI销售培训出题系统的角色和能力"""
        return """你是一个专业的企业销售培训出题系统，专精于销售技能提升领域的考题生成。
你拥有10年以上一线销售经验和销售管理经验，精通顾问式销售、SPIN销售法、解决方案销售、大客户销售等主流方法论。

你的核心能力是：针对同一个销售知识点，从不同维度生成不同的考题，确保每道题都考查不同的认知层面。

【出题原则】
1. 每个销售知识点必须从多个角度考查，不能只考死记硬背
2. 题目要能真实区分销售人员是否真正掌握了该技能，而不是"背答案"
3. 干扰项必须基于真实常见的销售误区和错误做法，不能随意编造
4. 案例题必须贴近真实销售场景，人物对话要自然真实
5. 一道题只考查一个核心销售知识点，不混淆多个考点

【多维度策略 - 销售培训版】
当你被要求从某个维度出题时，按以下思路设计：
- 概念理解(概念理解)：考查销售理论、方法论的定义、原理、适用场景的辨析
- 案例分析(案例分析)：给真实/模拟的销售案例，要求分析关键成功因素或失败原因
- 情景演练(情景演练)：给客户对话场景片段，考查话术应对能力和临场反应
- 异议处理(异议处理)：给客户的拒绝或异议场景，要求给出专业的处理方案
- 技巧对比(技巧对比)：比较不同销售技巧/方法在特定场景下的适用性和效果差异
- 场景选型(场景选型)：给定具体的业务场景（如客户类型、产品特点），选择最合适的销售策略
- 实战应用(实战应用)：将销售知识点放到实际工作场景中综合考查运用能力

【防抄袭要求】
- 题干表述要独特，避免与已有题目相似度超过40%
- 选择题的干扰项顺序必须随机打乱
- 如果涉及案例/对话，使用不同的客户姓名、产品名称、数据
- 数值类参数（如价格、折扣率）要使用不同的具体值

【输出要求】
你必须严格按照JSON格式输出，确保可以被程序正确解析。
不要在JSON前后添加任何markdown标记或额外文字。"""

    def build_user_prompt(self, context: GenerationContext) -> tuple:
        """
        根据上下文动态构造用户Prompt

        Returns:
            tuple: (prompt_text, selected_dimension) 返回构造好的Prompt和选中的维度
        """
        dimension = self._select_dimension(context)
        difficulty = self._select_difficulty(dimension)
        question_type = self._select_question_type(dimension)
        constraints = self._build_constraints(dimension, difficulty, question_type, context)
        variation_instruction = self._build_variation_instruction(context)
        excluded_summary = self._summarize_excluded(context.excluded_questions)

        prompt = f"""## 知识点
{context.knowledge_point}
{'（所属课程：' + context.course_name + '）' if context.course_name else ''}

## 出题约束
{constraints}

## 变体要求（防抄袭）
{variation_instruction}

## 已出题目摘要（请避免重复）
```
{excluded_summary}
```

## 输出格式
请严格输出以下JSON结构（纯JSON，不含markdown代码块标记）：

```json
{{
  "dimension": "{dimension.value}",
  "difficulty": "{difficulty.value}",
  "question_type": "{question_type.value}",
  "question_stem": "题干内容",
  "case_scenario": "案例/场景描述（如无则省略）",
  "options": ["选项A", "选项B", "选项C", "选项D"],
  "correct_answer": "正确答案",
  "explanation": "详细解析"
}}
```"""

        return prompt, dimension

    def _select_dimension(self, context: GenerationContext) -> QuestionDimension:
        """选择提问维度 - 优先选择未使用的维度"""
        available = [
            d for d in QuestionDimension
            if d.value not in context.excluded_dimensions
        ]
        if not available:
            return random.choice(list(QuestionDimension))
        return random.choice(available)

    def _select_difficulty(self, dimension: QuestionDimension) -> KnowledgeLevel:
        """根据维度选择合适的难度级别"""
        allowed = self.DIMENSION_DIFFICULTY_MAP.get(dimension, list(KnowledgeLevel))
        return random.choice(allowed)

    def _select_question_type(self, dimension: QuestionDimension) -> QuestionType:
        """根据维度选择合适的题型"""
        allowed = self.DIMENSION_TYPE_MAP.get(dimension, [QuestionType.SINGLE_CHOICE])
        return random.choice(allowed)

    def _build_constraints(
        self,
        dimension: QuestionDimension,
        difficulty: KnowledgeLevel,
        question_type: QuestionType,
        context: GenerationContext,
    ) -> str:
        """构建约束条件描述"""
        lines = [
            f"- 难度级别：{difficulty.value}",
            f"- 提问角度：{dimension.value}（{self._get_dimension_description(dimension)}）",
            f"- 题型：{question_type.value}",
        ]
        if context.special_requirements:
            lines.append(f"- 特殊要求：{context.special_requirements}")
        return "\n".join(lines)

    @staticmethod
    def _get_dimension_description(dimension: QuestionDimension) -> str:
        """获取维度的详细说明（销售培训版）"""
        descriptions = {
            QuestionDimension.CONCEPT: "考查对销售知识点的定义、原理、适用条件的理解程度",
            QuestionDimension.CASE_ANALYSIS: "给出真实/模拟销售案例，考查对关键信息的分析和判断能力",
            QuestionDimension.ROLE_PLAY: "给出客户对话场景片段，考查话术组织能力和临场应变能力",
            QuestionDimension.OBJECTION_HANDLING: "给出客户拒绝或异议的场景，考查异议处理的思路和方法",
            QuestionDimension.TECHNIQUE_COMPARISON: "比较不同销售方法/技巧在不同场景下的适用性和效果差异",
            QuestionDimension.SCENARIO_CHOICE: "给定具体业务场景（客户类型、产品、预算等），考查策略选型能力",
            QuestionDimension.PRACTICE_APPLICATION: "将销售知识点置于真实工作场景中综合考查运用能力",
        }
        return descriptions.get(dimension, "")

    def _build_variation_instruction(self, context: GenerationContext) -> str:
        """
        构建变体指令 - 这是防抄袭的核心
        
        基于学员ID生成确定性但差异化的变体要求，
        确保同一学员每次稳定，不同学员之间差异大
        """
        instructions = []
        if context.student_id:
            seed = int(hashlib.md5(context.student_id.encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            selected = rng.sample(self.VARIATION_STRATEGIES, k=min(2, len(self.VARIATION_STRATEGIES)))
            instructions.extend(selected)

        instructions.extend([
            "请确保题目与已出题目的题干表述、案例场景、提问方式至少有60%以上的差异",
            "选择题的干扰项顺序必须打乱，不能固定模式",
            "如果涉及案例/对话，使用不同的客户姓名、公司名、产品名、价格等具体信息",
            "正确答案的位置（A/B/C/D）要随机化",
        ])
        return "\n".join(f"- {inst}" for inst in instructions)

    @staticmethod
    def _summarize_excluded(excluded: List[str]) -> str:
        """对已出题目做摘要，防止Prompt过长导致token超限"""
        if not excluded:
            return "暂无已出题目"
        recent = excluded[-5:]
        lines = []
        for qid in recent:
            display_id = qid[:16] + "..." if len(qid) > 16 else qid
            lines.append(f"- 已出题ID: {display_id}")
        return "\n".join(lines) if lines else "暂无已出题目"
