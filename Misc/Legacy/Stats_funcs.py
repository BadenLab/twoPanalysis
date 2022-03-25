# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 20:39:22 2021

@author: skrem
"""

def shapirowilk(dataname, plot = 0, **kwargs):
    """Returns Shapiro-Wilk test for normality (stat, p-value), with the null hypothesis that
    data are normally distributed.
    
    Parameters
    ----------
    dataname: array-like
        Input data
    plot: 0 or 1
        1 plots histogram of sampled values and Q-Q plot for visual inspection
    kwargs:
        title = , hist_x = 
    """
    #for i in range(data):
        #generates probability plot for visual inspection of normality
    print("Null hypothesis: data are normally distributed")
    
    if isinstance(dataname, pd.Series) or isinstance(dataname, pd.DataFrame):
       dataname = dataname.dropna()  
    
    if plot == 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex =  False, sharey = False, figsize = (10, 5), dpi = 800)
        normality_plot, stat = scipy.stats.probplot(dataname, plot = ax2, rvalue= True)
        ax1.hist(dataname, color = 'black', histtype = 'stepfilled')
        ax2.get_lines()[0].set_marker('.')
        ax2.get_lines()[0].set_color('black')
        ax2.get_lines()[1].set_color('slategray')
        plt.subplots_adjust(wspace = .1)  
        ax1.set_title("Histogram")
        ax2.set_title("Q-Q plot")
        ax1.set_ylabel("Count")
        ax2.set_ylabel("Ordered values")
        if 'title' in kwargs:
            the_title = kwargs["title"]
            plt.suptitle(the_title)
        if 'hist_x' in kwargs:
            the_xlabel = kwargs["hist_x"]
            ax1.set_xlabel(the_xlabel)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.2, hspace=None)
        plt.show()

    #Runs Shapiro-Wilkns test for normality
    SW_res=scipy.stats.shapiro(dataname)
    print(SW_res)
    if SW_res[1] <= 0.05:
        print("Shapiro-Wilk test indicates rejection of null hypothesis: data are NOT normally distributed.")
    if SW_res[1] >= 0.05:
        print("Shapiro-Wilk test indicates null hypothesis is correct: data ARE normally distributed.")
    return SW_res

#_____________________________________________________________________________   
def Levenes(*sample_n): #assesses homogeneity of variances
    "Works best if passing DataFrames"
    lev_list = []
    for i in sample_n:
        if isinstance(i, pd.Series) or isinstance(i, pd.DataFrame):
           i_no_nan = i.dropna()
           lev_list.append(i_no_nan)
        else:
           lev_list.append(i)
    test = scipy.stats.levene(*lev_list)
    print(test)
    N = 0 #Number of cases in all groups
    k = 0 #Number of groups
    for i in lev_list:
        k += 1
        for j in i:
            N +=1
    levenes_dof = N-k
    print(f"Degrees of Freedom = {levenes_dof:.4f}")
    if test[1] < 0.05:
        print("Hypothesis of homogeniety of variance is rejected.")
    return test

#_____________________________________________________________________________
def WelchTtest(x,y):
    if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
        x = x.dropna()
    if isinstance(y, pd.Series)  or isinstance(y, y.DataFrame):
        y = y.dropna()
    Welchstat = scipy.stats.ttest_ind(x,y, equal_var  = False)
    print(Welchstat)
    welch_dof = (np.var(x)/x.size + np.var(y)/y.size)**2 / (np.var(x)/x.size)**2 / (x.size-1) + (np.var(y)/y.size)**2 / (y.size-1)
    print(f"Welch-Sattertwaite Degrees of Freedom = {welch_dof:.4f}")
#_____________________________________________________________________________
def StudentTtest(x,y):
    if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
        x = x.dropna()
    if isinstance(y, pd.Series)  or isinstance(y, y.DataFrame):
        y = y.dropna()
    Studstat = scipy.stats.ttest_ind(x,y, equal_var  = True)
    print(Studstat)
    stud_dof = x.size + y.size - 2
    print("Degrees of freedom df = nx+ny-2 =" ,stud_dof)
   
    #--> TO DO: Insert scipy.stats.ANOVA (or equivalent) for s to execute non/parametric test (EZ PZ)

#_____________________________________________________________________________
def kruskal(*sample_n):
    """a non-parametric method for testing whether samples originate from the same distribution
    """
    lev_list = []
    for i in sample_n:
        if isinstance(i, pd.Series) or isinstance(i, pd.DataFrame):
           i_no_nan = i.dropna()
           lev_list.append(i_no_nan)
        else:
           lev_list.append(i)
    result = scipy.stats.kruskal(*lev_list, nan_policy = 'omit')
    return result

def mann_whit(x, y, **kwargs):
    if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
        x = x.dropna()
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        y = y.dropna()
    if 'hypothesis' in kwargs:
        hypth = kwargs['hypothesis']
    else:
        hypth = 'two-sided'
    if 'cont_corr' in kwargs:
        cont_binary = kwargs['cont_corr']
    else:
        cont_binary = True
    basic_result = scipy.stats.mannwhitneyu(x, y, alternative = hypth, use_continuity = cont_binary)
    advanced_result = pingouin.mwu(x, y, tail = hypth)
    return basic_result, advanced_result
#_____________________________________________________________________________
def wilcoxon_SR(x, y, **kwargs):
    if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
        x = x.dropna()
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        y = y.dropna()
    "Non-parametric, paired samples"
    if 'method' in kwargs:
        method = kwargs['method']
    else:
        method = "wilcox"
    if 'correction' in kwargs:
        binary = kwargs['correction'] #True/false
    else:
        binary = False
    if 'alternative' in kwargs:
        alt = kwargs['alternative']
    else:
        alt = 'two-sided'
    if 'mode' in kwargs:
        mode = kwargs['mode']
    else:
        mode = 'auto'
    result = scipy.stats.wilcoxon(x, y, zero_method = method, correction = binary, alternative = alt, mode = mode)
    return result

#_____________________________________________________________________________
def wilcoxon_RS(x, y, **kwargs):
    if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
        x = x.dropna()
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        y = y.dropna()
    result = scipy.stats.ranksums(x, y)
    return result

def posthoc_dunn(x):
    scikit_posthocs.posthoc_dunn(x)
    

# condition_pallette = {"AZ": '#414288', "Peripheral": '#304A39', "AZ w CNQX": '#6A8D92', "Peripheral w CNQX": '#8DB99D'}
condition_pallette = {"AZ": '#274C77', "Peripheral": '#F85A3E', "AZ (CNQX)": '#5BC0EB', "Peripheral (CNQX)": '#D0768D'}

#____________________________________________________________________________
"Statistical plotting_________________________________________________________"
def boxplot(data_df, group_list, **kwargs):
    """Plots a boxplot. 
    
    Parameters
    -------------
    data_df: DataFrame
        Which df to plot from
    group_list: list
        List of columns in data_df to plot as groups
    **kwargs: 
        *x_label = (str) label of x axis
        *x_label = (str) label for y axis
        *title = (str) set suptitle 
        *sig_bar = (list of tuples) specify which groups to sig test and set bar for (and do prep stuff)
        *test_type = (str) choose: 'Mann-Whitney', 't-test_ind', 't-test_paired', 't-test_welch', 'Levene', 'Kruskal', 'Wilcoxon', 'Mann-Whitney-ls', 'Mann-Whitney-gt' 
        *'pvals' = (list) passes own statistics and skips test
    
    Example: boxplot(ellipses_areas, ['AZ', 'Peripheral', 'AZ w CNQX', 'Peripheral w CNQX'], x_label = 'x', y_label = 'y', plotpoints = 1, sig_bar = [("AZ", "Peripheral"), ("AZ", "AZ w CNQX")], test_type = 't-test_ind')"""
    plt.figure(dpi = 800)
    fig = sns.boxplot(data=data_df[group_list], palette = condition_pallette, linewidth = .75)
    if 'x_label' in kwargs:
        plt.xlabel(kwargs['x_label'])
    if 'y_label' in kwargs:
        plt.ylabel(kwargs['y_label'])
    if 'title' in kwargs:
        plt.suptitle(kwargs['title'])
    if 'sig_bar' in kwargs:
        pairs = kwargs['sig_bar'] #[(("AZ", "Peripheral"))]
        if 'test_type' in kwargs:
            test_type = kwargs['test_type'] #'t-test_ind'
            run_test = True
        else:
            test_type = None
        if 'pvals' in kwargs: 
            pvals = kwargs['pvals']
            run_test = False
        else:
            pvals = None
        if 'correction' in kwargs:
            corr = kwargs['correction']
            add_stat_annotation(fig, data = data_df[group_list], box_pairs = pairs, test = test_type, 
                            perform_stat_test = run_test, text_format = 'star', verbose=2, pvalues = pvals, comparisons_correction = corr)
        else:
            add_stat_annotation(fig, data = data_df[group_list], box_pairs = pairs, test = test_type, 
                            perform_stat_test = run_test, text_format = 'star', verbose=2, pvalues = pvals) #alternatively use text_format = 'simple'
    if 'plotpoints' in kwargs:
        sns.swarmplot(data=data_df[group_list], color = "w", edgecolor = 'gray', linewidth = 1 , size = 3)
    plt.grid()
    if 'showXaxis' in kwargs:
        if  kwargs['showXaxis'] == False or kwargs['showXaxis'] == 0:
            fig.axes.get_xaxis().set_visible(False)
    if 'saveas' in kwargs:
        plt.figure(dpi = 2000)
        plt.savefig(r'C://Users//skrem//OneDrive//Universitet//MSc//Experimental project//Figures//Python generated//{}'.format(kwargs['saveas']), dpi = 2000,  bbox_inches='tight')
    plt.show()
    
    "See: https://seaborn.pydata.org/tutorial/aesthetics.html"
    
def violinplot(data_df, group_list, **kwargs):
    """
    Parameters
    -------------
    data_df: DataFrame
        Which df to plot from
    group_list: list
        List of columns in data_df to plot as groups
    **kwargs: 
        *x_label = (str) label of x axis
        *x_label = (str) label for y axis
        *title = (str) set suptitle 
        *sig_bar = (list of tuples) specify which groups to sig test and set bar for (and do prep stuff)
        *test_type = (str) choose: 'Mann-Whitney', 't-test_ind', 't-test_paired', 't-test_welch', 'Levene', 'Kruskal', 'Wilcoxon', 'Mann-Whitney-ls', 'Mann-Whitney-gt' 
        *'pvals' = (list) passes own statistics and skips test

    """
    plt.figure(dpi = 800)
    if 'cut' in kwargs:
        fig = sns.violinplot(data=data_df[group_list], palette = condition_pallette, linewidth = .75, scale = "width", cut = kwargs['cut'])
    else:
        fig = sns.violinplot(data=data_df[group_list], palette = condition_pallette, linewidth = .75, scale = "count")
    if 'title' in kwargs:
        plt.suptitle(kwargs['title'])
    if 'x_label' in kwargs:
        plt.xlabel(kwargs['x_label'])
    if 'y_label' in kwargs:
        plt.ylabel(kwargs['y_label'])
    if 'sig_bar' in kwargs:
        pairs = kwargs['sig_bar'] #[(("AZ", "Peripheral"))]
        if 'test_type' in kwargs:
            test_type = kwargs['test_type'] #'t-test_ind'
            run_test = True
        else:
            test_type = None
        if 'pvals' in kwargs: 
            pvals = kwargs['pvals']
            run_test = False
        else:
            pvals = None
        add_stat_annotation(fig, data = data_df[group_list], box_pairs = pairs, test = test_type, 
                            perform_stat_test = run_test, text_format = 'star', verbose=2, pvalues = pvals) #alternatively use text_format = 'simple'
    plt.grid()

    if 'saveas' in kwargs:
        plt.savefig(r'C://Users//skrem//OneDrive//Universitet//MSc//Experimental project//Figures//Python generated//{}'.format(kwargs['saveas']), dpi = 2000,  bbox_inches='tight')
    plt.show()
    
"Data import/export___________________________________________________________"
Cond_order = ['AZ', 'AZ (CNQX)', 'Peripheral', 'Peripheral (CNQX)']
#_____________________________________________________________________________
def export_data(input_data, file_name, *col_label_list, path = r'C:\Users\skrem\OneDrive\Universitet\MSc\Experimental project\Data export'):
    
    make_df = pd.DataFrame.from_records(input_data)
    if make_df.shape[0] < make_df.shape[1]:
        make_df = make_df.transpose()
    
    if col_label_list:
        make_df.columns = col_label_list
    
    # create excel writer object
    data_name = file_name
    writer = pd.ExcelWriter(path + '\{}.xlsx'.format(data_name))
    # write dataframe to excel
    make_df.to_excel(writer)
    # save the excel
    writer.save()
    print('DataFrame is written successfully to Excel File.')
    
#_____________________________________________________________________________
def import_data(filename, original_formatting = 0, path = r'C:\Users\skrem\OneDrive\Universitet\MSc\Experimental project\Data export'):
    
    get_data = pd.read_excel(path + '\{}.xlsx'.format(filename))
    return_data = get_data.drop(get_data.columns[0], axis = 1)
    
    if original_formatting == 1:
        data_list = []
        for i in range(return_data.shape[1]):
            curr_list = return_data[return_data.columns[i]].tolist()
            data_list.append(curr_list)
        for n, j in enumerate(data_list):
            data_list[n] = [x for x in j if np.isnan(x) == False]            
        return pd.DataFrame(data_list).T
    else:
        return pd.DataFrame(return_data)

def sci_not(n):
    "Simple function for converting scientific notation to decimal-point. Forces str instead of float."
    decimal = ("%.17f" % n).rstrip('0').rstrip('.')
    # floated_decimal = float(n)
    print("Scientific notation: ", n)
    print("In decimal notation: ", decimal)
    return decimal