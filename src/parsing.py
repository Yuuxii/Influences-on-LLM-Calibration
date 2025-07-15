def get_float_prob(p):
    
    if ' ' in p:
        p = p.split(' ')[0]
    if p.endswith('.'):
        p = p[:-1]
    if p.endswith('%'):
        no_pct = p[:-1]
        p = str(float(no_pct) / 100)
    if p.startswith('<') and p.endswith('>'):
        p = p[1:-1]
    if p.startswith('*') and p.endswith('*'):
        p = p[1:-1]
    if p.startswith('*') and p.endswith('*'): # check again in case ** was used
        p = p[1:-1]
    
    return float(p)


# assuming answer is of the form "Guess: <guess> Probability: <probability>"
def parse_single_guess(answer, answer_prefix='Guess:'):
    a = 'FAILED TO PARSE'
    p = -1
    lines = answer.strip().split('\n')
    if len(lines) > 1:
        ans_lines = [l for l in lines if l.startswith(f'{answer_prefix}')]
        if len(ans_lines):
     
            a = ans_lines[0]
            if "Probability:" not in a:
                a = a[len(f'{answer_prefix}'):]
            else:
                a = a.split("Probability:")[0]

        else:
         
            ans_line_idxs = [i for i, l in enumerate(lines) if l.startswith(answer_prefix)]
            lines_contain_answer = [l for l in lines if answer_prefix in l]
            if len(ans_line_idxs):
           
                a = lines[ans_line_idxs[0] + 1]
            elif len(lines_contain_answer):
         
                a = lines_contain_answer[0]
                if "Probability:" not in a:
                    a = a.split(answer_prefix)[-1]
                else:
                    a = a.split(answer_prefix)[-1].split("Probability:")[0]
            else:
          
                a = lines[0]

        prob_lines = [l for l in lines if l.startswith('Probability:') or l.startswith('probability:')]
        if len(prob_lines):
            
            p = prob_lines[0]
            p = p[len('Probability:'):].strip()
        else:
            
            prob_line_idx = [i for i, l in enumerate(lines) if l.startswith('Probability:')]
            lines_contain_prob = [l for l in lines if 'Probability:' in l]
            if len(prob_line_idx):
                p = lines[prob_line_idx + 1]
            elif len(lines_contain_prob):
                
                p = lines_contain_prob[0]
                p = p.split('Probability:')[-1]
            else:
                
                p = lines[1]
    
    elif len(lines) == 1:
        if answer_prefix in lines[0]:
            ans = lines[0].split(answer_prefix)[-1]
                
            if "Probability:" in lines[0]:

                a = ans.split("Probability:")[0]
                p = ans.split("Probability:")[-1]
            else:
                p = -1
           
    # print('p:', p)
    return a.replace('\n', '').strip(), get_float_prob(str(p))


def parse_single_guess_safe(answer, answer_prefix='Guess:'):
    try:
        answer, confidence = parse_single_guess(answer, answer_prefix)
        # print("try", answer, confidence)
    except Exception as e:
        print(e)
        answer = 'FAILED TO PARSE'
        confidence = -1.0
    # print('confidence', confidence)
    return answer, confidence 





