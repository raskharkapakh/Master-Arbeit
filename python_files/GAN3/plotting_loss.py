import matplotlib.pyplot as plt


def plot_loss(history):
    
    title_fontsize = 20
    axis_fontsize = 17
    nb_fontsize = 15
    
    fig = plt.figure(figsize=(15,20))
    plt.subplot(211)
    plt.title('critic loss', fontsize=title_fontsize)
    plt.plot(history.history['c_loss'])
    plt.ylabel('loss', fontsize=axis_fontsize)
    plt.xlabel('epochs', fontsize=axis_fontsize)
    plt.xticks(fontsize=nb_fontsize)
    plt.yticks(fontsize=nb_fontsize)
    

    plt.subplot(212)
    plt.title('generator loss', fontsize=title_fontsize)
    plt.plot(history.history['g_loss'])
    plt.ylabel('loss', fontsize=axis_fontsize)
    plt.xlabel('epochs', fontsize=axis_fontsize)
    plt.xticks(fontsize=nb_fontsize)
    plt.yticks(fontsize=nb_fontsize)
    
    plt.show()