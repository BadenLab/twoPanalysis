#imports
#set plots to show inline
%matplotlib inline
#import necessary librarie
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from igor import binarywave
import seaborn as sns
from statistics import mean

from sklearn.preprocessing import normalize

#extract data
def extract(ibw_file):
    data = binarywave.load(ibw_file)
    file = data['wave']['wData']
    return file

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def RGBU1calc(Average, QI, stimulus_timing = 3, pause_timing = 6, nLEDs = 3, nBins = 20, QI_threshold = 0.5, eval_duration = 1, line_duration = 0.001956, include_rebount = 0, template_tau = 1):
    AllData = extract(Average) #Averages0
    AllQI = extract(QI) #QI

    nFrames = len(AllData)
    nROIs = len(AllData[0])

    nFr_per_stim = stimulus_timing/ line_duration
    nFr_per_pause = pause_timing/ line_duration
    nFr_per_eval = eval_duration/ line_duration #number of frames in the evaluation period (eg last n frames of stimulus)

    #Select cells above QI threshold
    nROIs_selected = 0
    selectedCellsAv = [] #selectedData
    selectedCellsQI = []
    selectedCells = []
    for count, selectedQI in enumerate(AllQI):
        if selectedQI > QI_threshold: #if QI of cell is above QI threshold
            selectedCellsAv.append(AllData[:,count]) #make list of arrays (average traces of each selected cell)
            selectedCellsQI.append(AllQI[count]) #list of QIs from selected cells
            selectedCells.append(count)
            nROIs_selected += 1
    selectedCellsAv = np.asarray(selectedCellsAv)
#     print('All Data: ', AllData[0,:])




# Baseline subtraction
# taking the most common brightness level as the baseline (we are assuming that cell spends most of its time at baseline)
# if give cell longer between stimuli can maybe just take first x seconds as baseline

    correctedCells = []
    Av = []
    for count, ROI in enumerate(selectedCellsAv):
        #define bins
        bins = np.arange(selectedCellsAv[count].min()-1, selectedCellsAv[count].max()+1, 0.2)
#         bins = np.arange(-50, 50, 0.2)  #TO GET TO LOOK THE SAME AS IGOR
        # histogram
        h, _ = np.histogram(selectedCellsAv[count], bins)
        # Find most frequent value
        baseline = bins[h.argmax()]
        for count2, frame in enumerate(ROI):
            selectedCellsAv[count, count2] -= baseline
#     print('slected data (corrected): ', selectedCellsAv[:,0:5])


    #Qualify responses by peak amplitudes: 3 colours, On and Off
    #get the mean of the average responses between the evaluation start and end time
    #for each LED for each ROI (24, 3)
    mean_responses = []
    for ROI_count, n in enumerate(selectedCellsAv):
        for LEDcount, LED in enumerate(range(nLEDs)):
            #get evaluation start and end frame for each stimulus
            ON_eval_start = int(((nFr_per_stim + nFr_per_pause) * (LEDcount+1)) - nFr_per_pause - nFr_per_eval)
            ON_eval_end = int(((nFr_per_stim + nFr_per_pause) * (LEDcount+1)) - nFr_per_pause)
            mean_responses.append(np.mean(selectedCellsAv[ROI_count, ON_eval_start : ON_eval_end]))
    mean_responses = np.reshape(np.asarray(mean_responses), (nROIs_selected,3))


#HAVEN@T INCLUDED REBOUNT HERE YET!!!!!!!!!!!!!!!!!!

    #RU index
    RUi=[]
    for ROI_count, n in enumerate(selectedCellsAv):
        UV = abs(mean_responses[ROI_count, 0])
        Red = abs(mean_responses[ROI_count, nLEDs-1])
        RUi.append((Red-UV)/(Red+UV))
#     print(RUi)
    RUi = np.asarray(RUi)

    #Plot
    #colour map
    colour = plt.get_cmap('gist_ncar_r')
    #restrict colour map
    new_cmap = truncate_colormap(colour, 0.15, 0.3)
    #Plot
    plt.figure(figsize=(7,5)) # Make it 14x7 inch
    plt.style.use('seaborn-whitegrid') # nice and clean grid
    plt.xlim(0,50)
    plt.xlabel('ROIs')
    plt.ylabel('Red-UV Index')
#     plt.title('Red-UV Index: ' + Average)
    #define bins
    bins = np.arange(-1, 1.05, 2/nBins)
    n, bins, patches = plt.hist(RUi, bins, orientation = 'horizontal', edgecolor='black', alpha = 0.6, linewidth = 1.5)
#     n, patches = plt.hist(RUi, bins, orientation = 'horizontal', edgecolor='black')
    sm=mpl.cm.ScalarMappable(cmap=new_cmap)
    # n = n.astype('int') # it MUST be integer
    # # Good old loop. Choose colormap of your taste
    for count, i in enumerate(range(len(patches))):
        patches[i].set_facecolor(sm.cmap(i/25))
        patches[i].set_edgecolor(sm.cmap(i/25))

    plt.savefig('RU.png') #saves to current folder
        
