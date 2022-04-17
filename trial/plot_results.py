import matplotlib.pyplot as plt
import pandas as pd
import validation_parser as vp

def plot(plt, loc_lst, lab_lst, x_metric = 'Steps', y_metric = 'BLEU-4', x_label = 'Steps', y_label = 'Bleu-4-Score'):
    assert len(loc_lst) == len(lab_lst), 'Location and Label should be of same length'
    for i in range(len(loc_lst)):
        df = vp.get_df(loc_lst[i])
        x_vec = df[x_metric]
        y_vec = df[y_metric]
        label = lab_lst[i]
        plt.plot(x_vec, y_vec, label = label)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    return plt

def plot_and_save(plt, loc_lst, lab_lst, save_loc = 'plot.png', x_metric = 'Steps', y_metric = 'BLEU-4', x_label = 'Steps', y_label = 'Bleu-4-Score'):
    plt = plot(plt, loc_lst, lab_lst, x_metric, y_metric, x_label, y_label)
    plt.savefig(save_loc)

if __name__ == '__main__':
    loc_lst = ['/home1/svadiraj/projects/DeepDive/slt/sign_sample_model/validations.txt', '/home1/svadiraj/projects/DeepDive/slt/sign_sample_model_convnext_large/validations.txt']
    lab_lst = ['Simple', 'Convnext']
    #plt = plot(plt, loc_lst, lab_lst)
    #plt.savefig('test.png')
    plot_and_save(plt, loc_lst, lab_lst, 'two_plots.png')