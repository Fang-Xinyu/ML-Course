from local_submission import submit as local_submit

def submit(question_id, student_answer):
    """
    本地评分系统接口，保持与原有 submit() 用法一致。
    :param question_id: 题目编号
    :param student_answer: 学生答案
    """
    return local_submit(question_id, student_answer)
