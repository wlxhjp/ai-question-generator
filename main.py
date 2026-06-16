"""
AI多维度出题系统 - 使用示例
基于LLM Prompt的智能出题引擎，支持同一知识点多角度变体生成

运行方式：
    1. 安装依赖：pip install openai pydantic
    2. 设置环境变量：export OPENAI_API_KEY="your-api-key"
    3. 运行：python main.py
"""

import os
from generator import create_generator, AIQuestionGenerator


def demo_single_question():
    """演示1：为单个学生生成一道题"""
    print("=" * 60)
    print("【演示1】为单个学生生成一道题")
    print("=" * 60)

    gen = create_generator(
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
        model="gpt-4o",
        temperature=0.8,
    )

    question = gen.generate_question(
        knowledge_point="Python列表推导式与生成器表达式的区别",
        student_id="stu_001",
        course_name="Python高级编程",
        special_requirements="出选择题，重点考查内存效率方面的理解",
    )

    _print_question(question)
    return question


def demo_batch_class():
    """演示2：为全班批量出题（防抄袭核心场景）"""
    print("")
    print("=" * 60)
    print("【演示2】为全班批量出题（防抄袭）")
    print("=" * 60)

    gen = create_generator(
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
        model="gpt-4o",
        temperature=0.8,
    )

    # 模拟一个班级的5个学生
    class_students = ["stu_001", "stu_002", "stu_003", "stu_004", "stu_005"]

    result = gen.batch_generate(
        knowledge_point="Python列表推导式与生成器表达式的区别",
        class_students=class_students,
        course_name="Python高级编程",
    )

    # 打印全班试卷分配情况
    print("")
    print(f"📊 出题统计：")
    print(f"   知识点：{result.knowledge_point}")
    print(f"   学生总数：{result.total_students}")
    print(f"   使用不同维度数：{result.unique_dimensions_used}")
    print(f"   生成不同题目数：{result.unique_questions_generated}")

    print(f"\n📋 全班试卷分配详情：")
    for sid, q in result.papers.items():
        print(f"   {sid}: 维度={q.dimension.value:6s} | "
              f"难度={q.difficulty.value:2s} | "
              f"题型={q.question_type.value:4s} | "
              f"ID={q.question_id}")

    # 验证防抄袭效果
    dimensions_used = [q.dimension.value for q in result.papers.values()]
    unique_dims = set(dimensions_used)
    print(f"\n✅ 防抄袭验证：")
    print(f"   全班使用了 {len(unique_dims)} 个不同的提问角度: {unique_dims}")

    question_ids = set(q.question_id for q in result.papers.values())
    print(f"   生成了 {len(question_ids)} 道不同的题目（无重复）")

    return result


def demo_multi_knowledge_points():
    """演示3：多个知识点的批量出题"""
    print("")
    print("=" * 60)
    print("【演示3】多知识点批量出题")
    print("=" * 60)

    gen = create_generator(
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
        model="gpt-4o",
        temperature=0.8,
    )

    knowledge_points = [
        "Python装饰器的原理与应用",
        "SQL索引的底层实现原理（B+树）",
        "Redis与MySQL数据一致性的解决方案",
        "微服务架构中的服务熔断机制",
    ]

    student_id = "stu_demo"

    for kp in knowledge_points:
        print(f"\n--- 知识点：{kp} ---")
        q = gen.generate_question(
            knowledge_point=kp,
            student_id=student_id,
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

    if question.code_snippet:
        print(f"\n   【代码片段】")
        for line in question.code_snippet.split("\n"):
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
    print("   方案B：基于LLM Prompt的多维度动态生成")
    print()

    # 运行各个演示（可根据需要注释/取消注释）
    
    try:
        # 演示1：单学生出题
        demo_single_question()

        # 演示2：全班批量出题（防抄袭）
        # demo_batch_class()

        # 演示3：多知识点出题
        # demo_multi_knowledge_points()

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        print("提示：请确保已设置 OPENAI_API_KEY 环境变量")
        print("   export OPENAI_API_KEY='your-api-key'")
