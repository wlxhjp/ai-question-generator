"""
AI多维度出题系统 - 使用示例（销售培训版）
基于LLM Prompt的智能出题引擎，支持同一知识点多角度变体生成

运行方式：
    1. 安装依赖：pip install openai pydantic python-dotenv
    2. 复制 .env 文件并填入你的 DeepSeek API Key
    3. 运行：python main.py
"""

import os
from dotenv import load_dotenv

# 加载 .env 环境变量文件（必须！否则 os.getenv 读不到 .env 中的值）
load_dotenv()

from generator import create_generator, AIQuestionGenerator


def demo_single_question():
    """演示1：为单个学员生成一道销售培训题"""
    print("=" * 60)
    print("【演示1】为单个学员生成一道销售培训题")
    print("=" * 60)

    gen = create_generator(
        model="deepseek-chat",
        temperature=0.8,
    )

    question = gen.generate_question(
        knowledge_point='SPIN提问法中的需求隐含问题（Implication Question）',
        student_id="trainee_001",
        course_name="顾问式销售实战训练",
        special_requirements="出单选题，重点考查如何通过隐含问题挖掘客户痛点",
    )

    _print_question(question)
    return question


def demo_batch_class():
    """演示2：为全班批量出题（防抄袭核心场景）"""
    print("")
    print("=" * 60)
    print("【演示2】为全班学员批量出题（防抄袭）")
    print("=" * 60)

    gen = create_generator(
        model="deepseek-chat",
        temperature=0.8,
    )

    class_trainees = ["trainee_001", "trainee_002", "trainee_003", "trainee_004", "trainee_005"]

    result = gen.batch_generate(
        knowledge_point='处理客户"价格太贵"异议的技巧与方法',
        class_students=class_trainees,
        course_name="销售异议处理专项训练",
    )

    print("")
    print(f"📊 出题统计：")
    print(f"   知识点：{result.knowledge_point}")
    print(f"   学员总数：{result.total_students}")
    print(f"   使用不同维度数：{result.unique_dimensions_used}")
    print(f"   生成不同题目数：{result.unique_questions_generated}")

    print(f"\n📋 全班试卷分配详情：")
    for sid, q in result.papers.items():
        print(f"   {sid}: 维度={q.dimension.value:6s} | "
              f"难度={q.difficulty.value:2s} | "
              f"题型={q.question_type.value:4s} | "
              f"ID={q.question_id}")

    dimensions_used = [q.dimension.value for q in result.papers.values()]
    unique_dims = set(dimensions_used)
    print(f"\n✅ 防抄袭验证：")
    print(f"   全班使用了 {len(unique_dims)} 个不同的提问角度: {unique_dims}")

    question_ids = set(q.question_id for q in result.papers.values())
    print(f"   生成了 {len(question_ids)} 道不同的题目（无重复）")

    return result


def demo_multi_knowledge_points():
    """演示3：多个销售知识点的批量出题"""
    print("")
    print("=" * 60)
    print("【演示3】多知识点批量出题（销售培训课程体系）")
    print("=" * 60)

    gen = create_generator(model="deepseek-chat", temperature=0.8)

    knowledge_points = [
        "客户画像与需求分析方法",
        "FABE法则在产品介绍中的应用",
        "大客户销售的信任建立策略",
        "成交信号识别与逼单时机把握",
    ]

    trainee_id = "trainee_demo"

    for kp in knowledge_points:
        print(f"\n--- 知识点：{kp} ---")
        q = gen.generate_question(
            knowledge_point=kp,
            student_id=trainee_id,
            excluded_questions=[],
            excluded_dimensions=[],
        )
        _print_question(q, verbose=False)


def _print_question(question, verbose=True):
    """格式化打印一道题目"""
    print(f"\n📝 题目信息：")
    print(f"   ID: {question.question_id}")
    print(f"   知识点: {question.knowledge_point}")
    print(f"   提问维度: {question.dimension.value}")
    print(f"   难度级别: {question.difficulty.value}")
    print(f"   题型: {question.question_type.value}")
    print(f"   试卷指纹: {question.paper_fingerprint}")

    print(f"\n   【题干】")
    print(f"   {question.stem}")

    if hasattr(question, 'case_scenario') and question.case_scenario:
        print(f"\n   【案例/场景】")
        for line in question.case_scenario.split("\n"):
            print(f"     {line}")

    if question.options:
        print(f"\n   【选项】")
        for i, opt in enumerate(question.options):
            marker = "✅ " if opt == question.correct_answer else "   "
            print(f"   {marker}{opt}")

    print(f"\n   【正确答案】: {question.correct_answer}")

    if verbose:
        print(f"\n   【解析】")
        for line in question.explanation.split("\n"):
            print(f"   {line}")


if __name__ == "__main__":
    print("🚀 AI多维度出题系统 - 使用示例")
    print("   方案B：基于LLM Prompt的多维度动态生成（DeepSeek + 销售培训版）")
    print()

    try:
        demo_single_question()
        # demo_batch_class()
        # demo_multi_knowledge_points()

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        print("\n请检查以下配置：")
        print("  1. 已安装依赖：pip install openai pydantic python-dotenv")
        print("  2. .env 文件中 DEEPSEEK_API_KEY 已填入真实 Key")
        print("     获取地址：https://platform.deepseek.com/")
        print("  3. 当前读取到的环境变量：")
        print(f"     DEEPSEEK_API_KEY = {os.getenv('DEEPSEEK_API_KEY', '❌ 未设置')[:20]}...")
        print(f"     DEEPSEEK_BASE_URL = {os.getenv('DEEPSEEK_BASE_URL', '❌ 未设置')}")
