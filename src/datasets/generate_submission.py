from src.utils.basic_utils import load_json, save_json
question_types = ['Interaction', 'Sequence', 'Prediction', 'Feasibility']


def trans_results(results, save_file):
    question_types = ['Interaction', 'Sequence', 'Prediction', 'Feasibility']
    submission = {t: [] for t in question_types}
    q_ids = []
    for result in results:
        q_id = result['question_id']
        if q_id in q_ids:
            continue
        q_ids.append(q_id)
        q_type = result['question_id'].split('_')[0]
        submission[q_type].append({'question_id': result['question_id'], 'answer': result['answer']})
    save_json(submission, save_file)








