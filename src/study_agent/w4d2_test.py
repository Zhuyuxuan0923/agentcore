"""Day 2 端到端测试：上传文档 + 提问"""

from study_agent.agent.kb_agent import PersonalQAAgent


def main():
    agent = PersonalQAAgent()

    # Step 1: 上传文档
    print("=== Step 1: 上传文档 ===")
    result = agent.upload("data/uploads/员工手册-测试.md", kb_name="员工手册")
    print(f"  文件: {result['file_name']}")
    print(f"  分块数: {result['chunk_count']}")

    # Step 2: 提问
    print("\n=== Step 2: 提问 ===")
    answer = agent.chat("年假怎么申请？有几天？")
    print(f"  问题: {answer.question}")
    print("  回答:")
    for line in answer.answer.split("\n"):
        print(f"    {line}")
    print(f"\n  引用数: {len(answer.citations)}")
    for c in answer.citations:
        snippet = c["text"][:80].replace("\n", " ")
        print(f"    [{c['number']}] {snippet}...")

    # Step 3: 再问一个
    print("\n=== Step 3: 再问一个 Wi-Fi ===")
    answer2 = agent.chat("公司 Wi-Fi 密码是什么？")
    print("  回答:")
    for line in answer2.answer.split("\n"):
        print(f"    {line}")

    print("\n[OK] 端到端测试完成!")


if __name__ == "__main__":
    main()
