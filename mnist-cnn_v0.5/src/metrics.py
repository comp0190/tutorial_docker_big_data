from sklearn.metrics import classification_report

def evaluate_model(y_true, y_pred, **kwargs):
    fcn_return_dict = lambda y_true, y_pred: classification_report(y_true, y_pred, output_dict=True)
    fcn_print = lambda y_true, y_pred: print(classification_report(y_true, y_pred))

    def fcn_save_file(y_true, y_pred):
        with open(kwargs['output_path'], 'w') as file:
            print(classification_report(y_true, y_pred), file=file)
        return 0

    output_type_dict = {
        'print': fcn_print,
        'return_dict': fcn_return_dict,
        'print2file': fcn_save_file,
    }

    return output_type_dict[kwargs['output_type']](y_true, y_pred)