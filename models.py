"""
AI多维度出题系统 - 数据模型定义
基于LLM Prompt的智能出题引擎，支持同一知识点多角度变体生成
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class KnowledgeLevel(str, Enum):
    """认知难度级别 - 基于布鲁姆认知分类学"""
    REMEMBER = "识记"
    UNDERSTAND = "理解"
    APPLY = "应用"
    ANALYZE = "分析"
    EVALUATE = "评价"


class QuestionDimension(str, Enum):
    """
    提问角度维度 - 同一知识点的不同考查方向
    
    核心防抄袭机制：不同学生从不同维度被提问，
    即使知识点相同，题目也完全不同
    """
    CONCEPT = "概念理解"              # 考查定义、原理、适用场景
    CODE_COMPLETION = "代码补全"      # 给不完整代码，填关键逻辑
    OUTPUT_PREDICTION = "输出预测"     # 给代码，问运行结果
    DEBUGGING = "错误排查"            # 给有bug的代码，找问题
    PERFORMANCE = "性能对比"          # 比较不同实现方式的效率
    DESIGN_CHOICE = "方案选型"        # 给需求场景，选技术方案
    SCENARIO_APPLICATION = "场景应用"  # 知识点放到实际业务场景


class QuestionType(str, Enum):
    """题型枚举"""
    SINGLE_CHOICE = "单选题"
    MULTIPLE_CHOICE = "多选题"
    JUDGE = "判断题"
    FILL_BLANK = "填空题"
    SHORT_ANSWER = "简答题"
    CODE_FILL = "代码填空题"
    DEBUG_FIX = "改错题"


class QuestionVariant(BaseModel):
    """
    生成的一道题变体
    
    每个变体通过以下维度实现差异化：
    - dimension: 提问角度（7种可选）
    - difficulty: 难度级别（5级）
    - stem: 题干表述（AI动态生成）
    - options: 干扰项（基于常见错误生成）
    - paper_fingerprint: 试卷指纹（用于追溯和防重复）
    """
    question_id: str = Field(description="题目唯一标识")
    knowledge_point: str = Field(description="所属知识点")
    dimension: QuestionDimension = Field(description="提问角度/维度")
    difficulty: KnowledgeLevel = Field(description="难度级别")
    question_type: QuestionType = Field(description="题型")
    stem: str = Field(description="题干内容")
    code_snippet: Optional[str] = Field(default=None, description="代码片段（可选）")
    options: Optional[List[str]] = Field(default=None, description="选项列表")
    correct_answer: str = Field(description="正确答案")
    explanation: str = Field(description="答案解析")
    paper_fingerprint: str = Field(description="试卷指纹（防抄袭追踪）")


class GenerationContext(BaseModel):
    """
    出题上下文 - 控制每次生成的变体方向
    
    核心作用：
    1. 记录已出题目，避免重复
    2. 记录已用维度，确保轮换
    3. 携带学生信息，实现个性化差异化
    """
    knowledge_point: str = Field(description="目标知识点")
    course_name: Optional[str] = Field(default=None, description="课程名称")
    excluded_questions: List[str] = Field(default_factory=list, description="已出题目的ID列表（防重复）")
    excluded_dimensions: List[str] = Field(default_factory=list, description="已使用的提问角度（确保轮换）")
    student_id: Optional[str] = Field(default=None, description="学生唯一标识（个性化种子）")
    class_id: Optional[str] = Field(default=None, description="班级ID（同班差异化）")
    special_requirements: Optional[str] = Field(default=None, description="特殊出题要求")


class BatchGenerationResult(BaseModel):
    """批量出题结果"""
    knowledge_point: str
    total_students: int
    unique_dimensions_used: int
    unique_questions_generated: int
    papers: dict = Field(default_factory=dict, description="{student_id: QuestionVariant}")


class TemplateConfig(BaseModel):
    """模板配置（方案A兼容结构）"""
    knowledge_point: str
    templates: List[dict]
    variable_pool: dict = Field(default_factory=dict)
