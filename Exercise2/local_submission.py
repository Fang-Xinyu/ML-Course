from answers import score

def submit(question_id, student_answer):
    result = score(question_id, student_answer)
    if result:
        print(f"题目 {question_id} 答对了！")
    else:
        print(f"题目 {question_id} 答错了，请再试一次。")
    return result
