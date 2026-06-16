"""
AI多维度出题系统 - Prompt构造器（方案B核心）
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
    多维度Prompt构造器 - 方案B的核心组件
    
    设计理念：
    1. 系统Prompt定义出题原则和质量标准
    2. 用户Prompt根据上下文动态构造，包含差异化指令
    3. 变体指令确保每个学生拿到不同的题目
    """

    # 维度到难度的映射：不同提问角度适合不同认知深度
    DIMENSION_DIFFICULTY_MAP = {
        QuestionDimension.CONCEPT: [KnowledgeLevel.REMEMBER, KnowledgeLevel.UNDERSTAND],
        QuestionDimension.CODE_COMPLETION: [KnowledgeLevel.UNDERSTAND, KnowledgeLevel.APPLY],
        QuestionDimension.OUTPUT_PREDICTION: [KnowledgeLevel.UNDERSTAND, KnowledgeLevel.APPLY],
        QuestionDimension.DEBUGGING: [KnowledgeLevel.APPLY, KnowledgeLevel.ANALYZE],
        QuestionDimension.PERFORMANCE: [KnowledgeLevel.ANALYZE, KnowledgeLevel.EVALUATE],
        QuestionDimension.DESIGN_CHOICE: [KnowledgeLevel.APPLY, KnowledgeLevel.ANALYZE, KnowledgeLevel.EVALUATE],
        QuestionDimension.SCENARIO_APPLICATION: [KnowledgeLevel.APPLY, KnowledgeLevel.ANALYZE],
    }

    # 维度到题型的映射
    DIMENSION_TYPE_MAP = {
        QuestionDimension.CONCEPT: [
            QuestionType.SINGLE_CHOICE,
            QuestionType.MULTIPLE_CHOICE,
            QuestionType.JUDGE,
            QuestionType.SHORT_ANSWER,
        ],
        QuestionDimension.CODE_COMPLETION: [QuestionType.CODE_FILL],
        QuestionDimension.OUTPUT_PREDICTION: [QuestionType.SINGLE_CHOICE, QuestionType.FILL_BLANK],
        QuestionDimension.DEBUGGING: [QuestionType.SINGLE_CHOICE, QuestionType.DEBUG_FIX],
        QuestionDimension.PERFORMANCE: [QuestionType.SINGLE_CHOICE, QuestionType.SHORT_ANSWER],
        QuestionDimension.DESIGN_CHOICE: [QuestionType.SINGLE_CHOICE, QuestionType.SHORT_ANSWER],
        QuestionDimension.SCENARIO_APPLICATION: [QuestionType.SINGLE_CHOICE, QuestionType.SHORT_ANSWER],
    }

    # 变体策略池 - 随机抽取2种来差异化每个学生的题目
    VARIATION_STRATEGIES = [
        "请使用不同的场景设定，避免与其他学生雷同",
        "请使用不同的数据示例和边界条件",
        "请从不同的角度切入该知识点",
        "请设计不同的代码示例，展示该知识点的不同侧面",
        "请设计含有常见误解的干扰项，基于不同的典型错误",
        "请将知识点放在不同的业务场景中考查",
        "请改变题干的表述方式，用不同的句式结构",
        "请调整选项的数量和排列顺序",
    ]

    @staticmethod
    def build_system_prompt() -> str:
        """构建系统级Prompt - 定义AI出题系统的角色和能力"""
        return """你是一个专业的教育出题系统，专精于IT技术领域的考题生成。
你的核心能力是：针对同一个知识点，从不同维度生成不同的题目，确保每道题都考查不同的认知层面。

【出题原则】
1. 每个知识点必须从多个角度考查，不能只考记忆
2. 题目要能真实区分学生是否理解，而不是"背答案"
3. 干扰项必须基于真实常见的错误认知，不能随意编造
4. 代码题必须给出可运行的代码片段，且格式规范
5. 一道题只考查一个核心知识点，不混淆多个考点

【多维度策略】
当你被要求从某个维度出题时，按以下思路设计：
- 概念理解(概念理解)：考查定义、原理、适用场景的辨析
- 代码补全(代码补全)：给不完整的代码，让学生填入关键逻辑
- 输出预测(输出预测)：给一段代码，问运行结果是什么
- 错误排查(错误排查)：给一段有bug的代码，让学生找出问题并修复
- 性能对比(性能对比)：比较不同实现方式的时间/空间效率差异
- 方案选型(方案选型)：给定需求场景，选择最合适的技术方案或工具
- 场景应用(场景应用)：把知识点放到实际业务场景中考查运用能力

【防抄袭要求】
- 题干表述要独特，避免与已有题目相似度超过40%
- 选择题的干扰项顺序必须随机打乱
- 如果涉及代码，使用不同的变量名和数据结构
- 数值类参数要使用不同的具体值和范围

【输出要求】
你必须严格按照JSON格式输出，确保可以被程序正确解析。
不要在JSON前后添加任何markdown标记或额外文字。"""

    def build_user_prompt(self, context: GenerationContext) -> tuple:
        """
        根据上下文动态构造用户Prompt
        
        Returns:
            tuple: (prompt_text, selected_dimension) 返回构造好的Prompt和选中的维度
        """
        # 1. 选择一个与已出题不同的维度（轮换机制）
        dimension = self._select_dimension(context)

        # 2. 根据维度确定难度和题型
        difficulty = self._select_difficulty(dimension)
        question_type = self._select_question_type(dimension)

        # 3. 构建约束条件文本
        constraints = self._build_constraints(dimension, difficulty, question_type, context)

        # 4. 构建变体指令（核心防抄袭逻辑）
        variation_instruction = self._build_variation_instruction(context)

        # 5. 构建已出题摘要
        excluded_summary = self._summarize_excluded(context.excluded_questions)

        # 6. 组装完整Prompt
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
  "code_snippet": "代码片段（如无则省略）",
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
            # 所有维度都用过了，重置为默认
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

        return "
".join(lines)

    @staticmethod
    def _get_dimension_description(dimension: QuestionDimension) -> str:
        """获取维度的详细说明"""
        descriptions = {
            QuestionDimension.CONCEPT: "考查对知识点的定义、原理、特征的理解程度",
            QuestionDimension.CODE_COMPLETION: "给出不完整代码片段，考查关键逻辑编写能力",
            QuestionDimension.OUTPUT_PREDICTION: "给出完整代码，考查对执行结果的预判能力",
            QuestionDimension.DEBUGGING: "给出含错误的代码，考查问题定位和修复能力",
            QuestionDimension.PERFORMANCE: "比较不同实现方式的效率差异",
            QuestionDimension.DESIGN_CHOICE: "给定需求场景，考查技术方案的选型能力",
            QuestionDimension.SCENARIO_APPLICATION: "将知识点置于真实业务场景中综合考查",
        }
        return descriptions.get(dimension, "")

    def _build_variation_instruction(self, context: GenerationContext) -> str:
        """
        构建变体指令 - 这是防抄袭的核心
        
        基于学生ID生成确定性但差异化的变体要求，
        确保同一学生每次稳定，不同学生之间差异大
        """
        instructions = []

        if context.student_id:
            # 用学生ID作为种子，确保同学生稳定、不同生差异
            seed = int(hashlib.md5(
                context.student_id.encode()
            ).hexdigest()[:8], 16)
            rng = random.Random(seed)

            # 从策略池中随机抽取2种变体策略
            selected = rng.sample(self.VARIATION_STRATEGIES, k=min(2, len(self.VARIATION_STRATEGIES)))
            instructions.extend(selected)

        # 通用防抄袭规则（所有学生都遵守）
        instructions.extend([
            "请确保题目与已出题目的题干表述、数据示例、提问方式至少有60%以上的差异",
            "选择题的干扰项顺序必须打乱，不能固定模式",
            "如果考代码，使用不同的变量名和数据结构",
            "正确答案的位置（A/B/C/D）要随机化",
        ])

        return "
".join(f"- {inst}" for inst in instructions)

    @staticmethod
    def _summarize_excluded(excluded: List[str]) -> str:
        """对已出题目做摘要，防止Prompt过长导致token超限"""
        if not excluded:
            return "暂无已出题目"
        
        # 在生产环境中，这里可以用embedding做语义去重摘要
        # 这里简化为返回最近5道的ID指纹
        recent = excluded[-5:]
        lines = []
        for qid in recent:
            display_id = qid[:16] + "..." if len(qid) > 16 else qid
            lines.append(f"- 已出题ID: {display_id}")
        return "
".join(lines) if lines else "暂无已出题目"
