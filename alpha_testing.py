import tldr

test_data = tldr.load_docs.load_docs("cnn/test/")

no_improvement = False
a = 1
increment = 0.1
prev_eval = 0

while(not no_improvement): # stop increasing alpha when theres no more improvement

    eval, scores = tldr.evaluate(test_data,
                            stem_tokens=True,
                            case_normalize=True,
                            open_class_tagging=True,
                            inverted_pyramid=True,
                            log_weight=True,
                            alpha=a)

    if eval <= prev_eval: # check if improvement
        no_improvement = True
        a -= increment # revert back to previous alpha

    else:
        a += increment # if improvement, increment alpha and run again

print("Best alpha found: ", a)