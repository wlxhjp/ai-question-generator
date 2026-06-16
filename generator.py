"""
AI多维度出题系统 - 核心出题引擎（方案B完整实现）
基于LLM Prompt的智能出题引擎，支持同一知识点多角度变体生成
"""

import json
import re
import time
import hashlib
import random
from typing import List, Optional, Dict

from openai import OpenAI

from models import (
    KnowledgeLevel,
    QuestionDimension,
    QuestionType,
    QuestionVariant,
    GenerationContext,
    BatchGenerationResult,
)
from prompt_builder import PromptBuilder


class AIQuestionGenerator:
    """
    AI多维度出题引擎 - 方案B的完整实现
    
    核心能力：
    1. 为单个学生生成个性化题目变体
    2. 为全班批量出题（防抄袭）
    3. 自动重试 + 降级兜底机制
    
    防抄袭原理：
    - 不同学生 → 不同种子 → 不同变体策略组合
    - 同班学生 → 维度轮换 → 提问角度不同
    - 选项乱序 + 干扰项随机化 → 即使题干相似答案也不同
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.8,
        max_retries: int = 3,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.base_temperature = temperature
        self.max_retries = max_retries
        self.prompt_builder = PromptBuilder()

    def generate_question(
        self,
        knowledge_point: str,
        student_id: Optional[str] = None,
        class_id: Optional[str] = None,
        course_name: Optional[str] = None,
        excluded_questions: Optional[List[str]] = None,
        excluded_dimensions: Optional[List[str]] = None,
        special_requirements: Optional[str] = None,
    ) -> QuestionVariant:
        """
        为指定学生生成一道多维度变体题目（核心方法）

        Args:
            knowledge_point: 知识点名称（如"Python列表推导式"）
            student_id: 学生唯一标识，用于生成个性化变体
            class_id: 班级ID，同班学生差异化
            course_name: 课程名称
            excluded_questions: 已出题目的ID列表，避免重复
            excluded_dimensions: 已使用的提问角度，确保轮换
            special_requirements: 特殊出题要求

        Returns:
            QuestionVariant: 生成的题目变体
        """
        context = GenerationContext(
            knowledge_point=knowledge_point,
            course_name=course_name,
            excluded_questions=excluded_questions or [],
            excluded_dimensions=excluded_dimensions or [],
            student_id=student_id,
            class_id=class_id,
            special_requirements=special_requirements,
        )

        for attempt in range(self.max_retries):
            try:
                # 1. 构造Prompt
                system_prompt = self.prompt_builder.build_system_prompt()
                user_prompt, selected_dimension = self.prompt_builder.build_user_prompt(context)

                # 2. 调用LLM（温度递增以增加多样性）
                current_temp = self.base_temperature + (attempt * 0.05)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=current_temp,
                    response_format={"type": "json_object"},
                )

                # 3. 解析响应
                raw_content = response.choices[0].message.content.strip()
                parsed = self._parse_response(raw_content)

                # 4. 后处理校验与修复
                validated = self._validate_and_fix(parsed, context, selected_dimension)
                if validated:
                    return validated

                print(f"[Attempt {attempt + 1}/{self.max_retries}] 校验失败，重试中...")

            except Exception as e:
                print(f"[Attempt {attempt + 1}/{self.max_retries}] 异常: {e}")
                if attempt == self.max_retries - 1:
                    return self._fallback_question(knowledge_point, context)

            time.sleep(0.5)  # 避免API限流

        # 所有重试都失败，降级为模板兜底
        return self._fallback_question(knowledge_point, context)

    def batch_generate(
        self,
        knowledge_point: str,
        class_students: List[str],
        course_name: Optional[str] = None,
        **kwargs,
    ) -> BatchGenerationResult:
        """
        为一个班级的学生批量出题（防抄袭核心方法）

        策略：
        1. 每个学生独立调用 generate_question
        2. 通过 excluded_dimensions 实现维度轮换
        3. 确保同班学生的提问角度尽可能不重复

        Args:
            knowledge_point: 知识点
            class_students: 学生ID列表
            course_name: 课程名称
            **kwargs: 其他传递给 generate_question 的参数

        Returns:
            BatchGenerationResult: 批量出题结果
        """
        papers: Dict[str, QuestionVariant] = {}
        used_dimensions: List[str] = []

        for student_id in class_students:
            question = self.generate_question(
                knowledge_point=knowledge_point,
                student_id=student_id,
                course_name=course_name,
                excluded_dimensions=list(used_dimensions),  # 传递已用维度
                **kwargs,
            )
            papers[student_id] = question
            used_dimensions.append(question.dimension.value)

        # 统计结果
        unique_dims = len(set(q.dimension.value for q in papers.values()))
        unique_qs = len(set(q.question_id for q in papers.values()))

        return BatchGenerationResult(
            knowledge_point=knowledge_point,
            total_students=len(class_students),
            unique_dimensions_used=unique_dims,
            unique_questions_generated=unique_qs,
            papers=papers,
        )

    # ==================== 内部方法 ====================

    @staticmethod
    def _compute_seed(
        student_id: Optional[str],
        class_id: Optional[str],
        knowledge_point: str,
    ) -> int:
        """计算确定性种子值
        
        用途：
        - 同一学生每天拿到稳定题目（方便续考/重考）
        - 不同学生拿到不同变体
        - 同班学生的参数差异最大化
        """
        seed_str = f"{student_id or 'anonymous'}_{class_id or 'default'}_{knowledge_point}"
        seed_str += f"_{time.strftime('%Y%m%d')}"
        return int(hashlib.sha256(seed_str.encode()).hexdigest()[:16], 16)

    @staticmethod
    def _parse_response(raw: str) -> Optional[dict]:
        """解析LLM返回的JSON字符串"""
        # 清理可能的markdown代码块标记
        cleaned = re.sub(r'^```(?:json)?\s*', '', raw.strip())
        cleaned = re.sub(r'\s*```$', '', cleaned)
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # 尝试提取第一个完整的JSON对象
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    return None
            return None

    def _validate_and_fix(
        self,
        parsed: Optional[dict],
        context: GenerationContext,
        dimension: QuestionDimension,
    ) -> Optional[QuestionVariant]:
        """验证LLM输出完整性并进行必要修复"""
        if not parsed:
            return None

        # 检查必填字段
        required_fields = ["question_stem", "correct_answer", "explanation"]
        for field in required_fields:
            if not parsed.get(field):
                print(f"校验失败：缺少必填字段 '{field}'")
                return None

        # 生成唯一ID和试卷指纹
        fingerprint_raw = (
            f"{context.knowledge_point}_"
            f"{parsed['question_stem'][:20]}_"
            f"{time.time()}"
        )
        fingerprint = hashlib.md5(fingerprint_raw.encode()).hexdigest()[:8]

        # 解析枚举值（容错处理）
        dim_value = parsed.get("dimension", dimension.value)
        resolved_dim = self._resolve_enum(QuestionDimension, dim_value, QuestionDimension.CONCEPT)

        diff_value = parsed.get("difficulty", "理解")
        resolved_diff = self._resolve_enum(KnowledgeLevel, diff_value, KnowledgeLevel.UNDERSTAND)

        type_value = parsed.get("question_type", "单选题")
        resolved_type = self._resolve_enum(QuestionType, type_value, QuestionType.SINGLE_CHOICE)

        return QuestionVariant(
            question_id=f"q_{fingerprint}",
            knowledge_point=context.knowledge_point,
            dimension=resolved_dim,
            difficulty=resolved_diff,
            question_type=resolved_type,
            stem=parsed["question_stem"],
            code_snippet=parsed.get("code_snippet"),
            options=parsed.get("options"),
            correct_answer=parsed["correct_answer"],
            explanation=parsed["explanation"],
            paper_fingerprint=fingerprint,
        )

    @staticmethod
    def _resolve_enum(enum_class, value: str, default):
        """安全地解析枚举值，解析失败返回默认值"""
        for item in enum_class:
            if item.value == value or item.name == value:
                return item
        # 模糊匹配
        for item in enum_class:
            if value in item.value or item.value in value:
                return item
        return default

    def _fallback_question(
        self,
        knowledge_point: str,
        context: GenerationContext,
    ) -> QuestionVariant:
        """
        LLM调用失败时的降级方案
        
        生产环境中可替换为：
        1. 从预置模板库中选取
        2. 返回缓存的历史题目
        3. 使用规则引擎生成基础题目
        """
        ts = int(time.time())
        return QuestionVariant(
            question_id=f"fallback_{ts}",
            knowledge_point=knowledge_point,
            dimension=QuestionDimension.CONCEPT,
            difficulty=KnowledgeLevel.REMEMBER,
            question_type=QuestionType.SINGLE_CHOICE,
            stem=f"关于「{knowledge_point}」，以下哪个描述是正确的？",
            options=[
                "A. （此处应为正确描述）",
                "B. （此处应为基于常见误解的错误描述）",
                "C. （此处应为混淆概念的错误描述）",
                "D. （此处应为无关内容的错误描述）",
            ],
            correct_answer="A",
            explanation="⚠️ 此题为降级模式生成的占位题目。请检查API配置后重新生成。",
            paper_fingerprint=f"fallback_{context.student_id}_{ts}",
        )


# ==================== 工厂方法 ====================

def create_generator(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o",
    temperature: float = 0.8,
) -> AIQuestionGenerator:
    """
    工厂方法：创建出题引擎实例
    
    Args:
        api_key: OpenAI API Key。如未提供则从环境变量 OPENAI_API_KEY 读取
        base_url: API基础URL（用于兼容其他OpenAI兼容服务）
        model: 模型名称
        temperature: 生成温度（越高越多样）

    Returns:
        AIQuestionGenerator: 出题引擎实例
    """
    import os

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "未提供API Key。请通过参数传入或设置环境变量 OPENAI_API_KEY"
        )

    return AIQuestionGenerator(
        api_key=key,
        base_url=base_url,
        model=model,
        temperature=temperature,
    )
