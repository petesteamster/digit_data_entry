def get_val_from_dict_csv(t_key):
    #input is a key. That is the row entry key in the dictionary_inputs.csv file.
    #Read the file into pandas. Convert the Name column into a list. Find the
    #index where the lest entry = the_key. Use the index to get the value
    #from a list made from th Value column in the file. Return the value as
    #a string
    import pandas as pd
    t_prefix=set_file_name_prefix(in_colab())[0]
    t_file=t_prefix+'dictionary_inputs.csv'
    df=pd.read_csv(t_file)
    t_list_n=list(df['Name'])
    t_pos=t_list_n.index(t_key)
    t_list_v=list(df['Value'])
    return str(t_list_v[t_pos])
def update_class_number(nw_class_index):
    #reads in dictionary_inputs.csv which holds values for global
    #variable where the name is the global variable name. Once read in,
    #it updates class label. This needs to be done via user so it can be changed
    #as a user enters different digit classes. 
    import pandas as pd
    t_list02=[]
    for i in range(0,10):
        t_class='?class0'+str(i)+'?'
        t_list02.append(t_class)
    nw_class=t_list02[nw_class_index]    
    t_prefix=set_file_name_prefix(in_colab())[0]
    t_file=t_prefix+'dictionary_inputs.csv'
    df=pd.read_csv(t_file)
    t_list=list(df['Name'])
    t_pos=t_list.index('t_class')
    df.at[t_pos, 'Value']=nw_class
    t_pos=t_list.index('?selval?')
    df.at[t_pos, 'Value']=nw_class_index
    df.to_csv(t_file,index=False)
    return df
def get_class_counts(t_file):
    import pandas as pd
    #gets a dictionary with the counts of classes that are in 
    #digit_data.csv. If the file does not exist, it returns a dictioary
    #with only a single entry with 0 for count. This is how the
    #situation where no digit_data.csv exists is handled. Groupby 
    #returns a dictionary
    dd={'replace_dict': {'?class02?': 0}}
    fname_pfx=set_file_name_prefix(0)[0]
    #print(fname_pfx)
    if(exist_digit_data(t_file)==1):
        df=pd.read_csv(fname_pfx+t_file,header=None)
        #print(df.shape)
        t_df=df.iloc[:,0:2]
        t_list=['class','replace_dict']
        t_df.columns = t_list
        dd=t_df.groupby('class').count().to_dict()
    return dd
def exist_digit_data(t_file):
    #checks to see if digit_data.csv exist. It hold the classification
    # labels and the vectors. It is also when counting the number
    # of each individual class. 
    t_prefix=set_file_name_prefix(0)[0]
    from os import path
    t_file_name=set_file_name_prefix(0)[0]+t_file
    #print(t_file_name)
    return int(path.exists(t_file_name))       
def insert_mods_to_js_file(t_js_file,the_dict):
    #This function takes as input a js file and a dict.
    #The js file has wild card values for certain varibles 
    #And functions. These wildcards get swapped for values
    #cross referenced in the dictionary that is also input.
    #This allows the python to communicate with the javascript.
    from digit_capture import set_file_name_prefix,in_colab
    from digit_capture import read_pickle,write_pickle
    
    fname_pfx=set_file_name_prefix(0)[0]
    #print(fname_pfx)
    #pk_file=t_js_file[0,-3]+'.pk'
    #the_dict=read_pickle(fname_pfx+pk_file)
    #file1 = open(fname_pfx+t_js_file,"r+")
    file1 = open(fname_pfx+t_js_file,"r+")  
    the_js_data=file1.read()
    for old,new in the_dict.items():
        the_js_data=the_js_data.replace(old,str(new))
    return the_js_data
#insert_mods_to_js_file(t_js_file,t_dict)
def cross_ref_replace_dict(t_file_name):
    #This function creates the cross reference dictionary used by
    #insert_mods_into_js_file
    import pandas as pd
    from digit_capture import set_file_name_prefix
    from digit_capture import get_global_settings as gs
    colab_choice_save=['not_colab_output(imgData)','colab_output(imgData)']
    colab_choice_label=['change_class_label_no_colab()',
                        'change_class_label_colab()']
    t_full_side_size=gs('grid_count')*gs('pxl')
    cross_ref_dict={'?width?':t_full_side_size,'?height?':t_full_side_size}
    test_keys = ["?class00?",
                 "?class01?",
                 "?class02?",
                 "?class03?",
                 "?class04?",
                 "?class05?",
                 "?class06?",
                 "?class07?",
                 "?class08?",
                 "?class09?",
                 "?selval?",
                 "?width?",
                 "?height?"]
    test_values = [0,0,0,0,0,0,0,0,0,0,3,t_full_side_size,t_full_side_size] 
    # using dictionary comprehension 
    # to convert lists to dictionary 
    res = {test_keys[i]: test_values[i] for i in range(len(test_keys))}
    res['?selval?']=get_val_from_dict_csv('?selval?')
    #fname_pfx=set_file_name_prefix(0)[0]
    #print(fname_pfx)
    #df=pd.read_csv(fname_pfx+t_file_name,header=None)
    #t_df=df.iloc[:,0:2]
    #t_list=['class','replace_dict']
    #t_df.columns = t_list
    #dd=t_df.groupby('class').count().to_dict()
    dd=get_class_counts(t_file_name)
    #print(dd['replace_dict'])
    for k, v in dd['replace_dict'].items():
        #print(k,dd['replace_dict'][k])
        res[k]=dd['replace_dict'][k]
    res['?grid_count?']=get_dict()['grid_count']
    res['?pxl?']=get_dict()['pxl']
    res['?line_wd?']=get_dict()['line_wd']
    res['?output_data?']=colab_choice_save[in_colab()]
    res['?class_label_choice?']=colab_choice_label[in_colab()]
    return res
def set_correct_type(t_values,t_types):
    #Converts the list holding values for the global dictionary
    #into integers if that is the desired type based on the list 
    #t_types
    t_len=len(t_values)
    new_list=[]
    for i in range(0,t_len):
        nw_val=t_values[i]
        if t_types[i]=='int':
            nw_val=int(nw_val)
        new_list.append(nw_val)
    return new_list    
def get_dict_from_file(t_file):
    #reads in a dataframe and converts it into a dictionary.
    #The dictioary holds a lot of values of global variables.
    from digit_capture import set_file_name_prefix
    import pandas as pd
    fname_pfx=set_file_name_prefix(0)[0]
    #print(fname_pfx)
    df=pd.read_csv(fname_pfx+t_file)
    t_name=list(df['Name'])
    t_value=list(df['Value'])
    t_type=list(df['Type'])
    t_value=set_correct_type(t_value,t_type)
    res = {t_name[i]: t_value[i] for i in range(len(t_name))} 
    return res
def get_dict():
    #dictionary holds the values of global variables.
    gbl=get_dict_from_file('dictionary_inputs.csv')
#     gbl={'grid_count':5,
#          'line_wd':5,
#          'pxl':50,
#          'd_2':50,
#          'final_side_size':28,
#          't_class':'?class02?',
#          'min_pxl_count':5,
#          'threshold_low':20
#         }
    return gbl
def get_global_settings(t_var):
        #used to look up the values of global variables 
    gbl=get_dict()
    rtn=gbl[t_var]
    return rtn
def mount_drive():
    #Used to mount google drive when run in colab
    if(not in_colab()):
        return
    from google.colab import drive
    drive.mount('/content/drive')
def in_colab():
    #return '1' if run in colab, '0' if not run in colab
    import sys
    IN_COLAB = 'google.colab' in sys.modules
    return int(IN_COLAB)
def split_target_normalize(t_data,t_column):
    import pandas as pd
    import numpy as np
    t_data_features=t_data.drop(columns=[t_column])
    t_data_features = np.ceil(t_data_features/255)
    t_labels=t_data[[t_column]]
    rtn=pd.concat([t_labels,t_data_features],axis=1)
    return rtn
        
def set_file_name_prefix(colab):
    # Gets pathname etc based on there it is run.
    # if colab=1, set up structure for colab.
    # if colab=0, set up structure for local.
    # if colab=2, set up structure when in /content/drive  
    import sys
    import os
    t_prefix=""
    t_path=""
    if(in_colab()==1):
        colab=1
    if(colab==1):
        t_path='/content/drive/My Drive/Colab Notebooks/digit_data_entry/digit_data_entry'
        t_prefix=t_path+"/"
    else:
        if(colab==0):
            t_prefix="/content/drive"
        st=os.getcwd()    
        t_path=st+"/"+t_prefix
        t_prefix=t_path+"/"
    return (t_prefix,t_path)    
def restartkernel():
    from IPython.display import display_html
    display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)
def separate_digits(t_image,t_dim,grid_count):
    #Takes in the image drawn on the canvas and cuts each grid square out,
    #Returns a list of the separate grids and a 3D array with an
    #entry for each grid square
    import numpy as np
    import cv2
    from digit_capture import get_global_settings as gs
    final_side_size=gs('final_side_size')
    grid_count=gs('grid_count')
    row_count=grid_count*grid_count
    #print("separate "+str(t_image.shape))
    #print("row_count "+row_count)
    #print("final_side_size "+final_side_size)
    im_list=[]
    im_reshaped=np.zeros((row_count,final_side_size,final_side_size))
    dim_list=[]
    for i in range(0,grid_count):
        dim_list.append((i+1)*t_dim)    
    st_1=0
    st_2=0
    t_index=0
    for i in dim_list:
        for j in dim_list:
            t_d=t_image[st_1:i,st_2:j]
            if(check_if_enough_pixels(t_d)==1):
                t_rs=cv2.resize(t_d,(final_side_size,final_side_size))
                im_list.append(t_d)
                im_reshaped[t_index,]=t_rs
                t_index=t_index+1
            st_2=st_2+t_dim
        st_1=st_1+t_dim
        st_2=0
    return (im_list,im_reshaped[0:t_index,])
def check_if_enough_pixels(t_image):
      # The function counts the number of pixels in the figure.
      # It returns '1' if there are enough pixels, '0' if not enough.
      # Enough defined as greater than min_pxl_count in global_settings
      # A canvas with no figure will have no 'figure pixels'
    import cv2
    import numpy as np
    from digit_capture import get_global_settings as gs 
    (thresh,gray) = cv2.threshold(t_image,gs('threshold_low'),255,cv2.THRESH_BINARY)
    gray=255-gray
    t_sum=np.sum(gray)/255
    #print('t_sum '+str(t_sum))
    #print(gray)
    rtn=0
    if(t_sum>gs('min_pxl_count')):
        rtn=1
    return rtn                        
def save(t_canvas_image):
    #function used to save imaages to a csc file.
    #It takes in the full canvas. It separates the grid squares
    # (via separate_digits), refromats them (via reformat_mnist) and
    # reshapes then into a one dimensional vector that is written to
    # a csv file that can be used as input to different machine
    # learning programs. The output includes a class label. The label
    # has the same value for all the digits on the filled out grid sqaures.
    # Only one class at a time can be entered for each set of grid squares
    import pandas as pd
    import cv2
    import numpy as np
    from digit_capture import get_global_settings as gs
    d_2=gs('d_2')
    grid_count=gs('grid_count')
    t_dest=set_file_name_prefix(0)
    fname_pfx=t_dest[0]
    t_path=t_dest[1]
    t_class=gs('t_class')
    final_side_size=gs('final_side_size')
    df_class=make_class_list(t_class,(grid_count*grid_count))
    df_column_names=make_col_list((final_side_size*final_side_size))
    out_array = np.zeros(((grid_count*grid_count),(d_2*d_2)),dtype='uint8')
    (t_digits,t_reformat_dg)=separate_digits(t_canvas_image,d_2,grid_count)
    new_data=reformat_mnist(t_reformat_dg,0)
    df_dta=pd.DataFrame(data=new_data)
    df_dta=pd.concat([df_class,df_dta],axis=1)
    write_digit_data(df_dta,fname_pfx+'digit_data.csv')
#     for i in range(0,(grid_count*grid_count)):
#         new_out=np.reshape(new_data[i,:],(final_side_size,final_side_size))
#         t_ind=str(i).zfill(5)
#         cv2.imwrite(fname_pfx+'digit'+t_ind+'.jpg',new_out)
    #print(len(t_digits))
def write_digit_data(t_data,file_name):
    import pandas as pd
    t_data.to_csv(file_name,header=False,mode='a',index=False)
def make_class_list(t_class,class_count):
    #This returns a class label entry for each row (one dim vector made
    #from the image). 
    import pandas as pd
    lst=[t_class]*class_count
    df=pd.DataFrame(lst)
    return df                    
def do_pca(t_data):
    #Used for setting up a PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn import metrics
    import pandas as pd
    scaler = StandardScaler()
    pca = PCA(.999)
    # Fit on training set only.
    scaler.fit(t_data)
    # Apply transform to both the training set and the test set.
    train_data = scaler.transform(t_data)
    pca.fit(train_data)
    #print(pca.n_components_)
    train_data = pca.transform(train_data)
    return train_data
def write_pickle(file_name,obj_name):
    #Used to write a pickle file
    import pickle as pk
    #print(file_name)
    #with open(file_name,'wb') as f:
    #    pk.dump(obj_name,f)
    pk.dump( obj_name, open( file_name, "wb" ) )
def read_pickle(file_name):
    #used to read a pickle file
    import pickle as pk
    rtn=0
    with open(file_name,'rb') as f:
        rtn=pk.load(f)
    return rtn    
def getBestShift(img):
    #Used to help center the image. The goal is to
    #align the images.
    import numpy as np
    import scipy.ndimage as ndi
    from digit_capture import get_global_settings as gs
    fss=gs('final_side_size')
    cy,cx = ndi.measurements.center_of_mass(255-img)
    shiftx = np.round(fss/2.0-cx).astype(int)
    shifty = np.round(fss/2.0-cy).astype(int)
    return shiftx,shifty
def add_col_list(t_data):
    r_c=t_data.shape
    col_list=make_col_list(r_c[1])
    t_data.columns=col_list
    return t_data
def make_col_list(t_count):
    #This is used to create column names for each entry in the single dimension
    #vector created from the image
    rtn=[]
    rtn.append('t_class')
    for i in range(1,t_count):
        rtn.append('col_'+str(i).zfill(4))
    return rtn
def reshape_mnist(tData):
    import numpy as np
    from digit_capture import get_global_settings as gs
    fss=gs('final_side_size')
    entries=tData.shape[0]
    new_format=np.zeros((entries,(fss*fss)),dtype='float')
    for i in range(0,entries):
        (thresh, gray) = cv2.threshold(tData[i,:,:],gs('threshold_low'), 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        new_format[i,:]=np.reshape(gray,(1,fss*fss))
    return new_format    
def reformat_mnist(tData,t_invert):
    # Finds the bounding square ff the image and
    # then centers the the image in a 28 by 28 field
    import numpy as np
    from digit_capture import get_global_settings as gs
    fss=gs('final_side_size')
    entries=tData.shape[0]
    new_format=np.zeros((entries,(fss*fss)),dtype='uint8')
    for i in range(0,entries):
        t_image=set_in_bounding_square(tData[i,:,:],20,i)
        shifted=shift_2(t_image,t_invert)
        shifted=np.reshape(shifted,(fss*fss,))
        new_format[i,:]=shifted
    return new_format    
def load_and_threshold_file(t_file):
    #Reads in an image file and thresholds it. 
    import cv2
    import numpy as np
    from digit_capture import get_global_settings as gs
    fss=gs('final_side_size')
    gray1=cv2.imread(t_file,0)
    gray=cv2.imread(t_file,0)
    gray = cv2.resize(gray,(fss, fss))
    (thresh, gray) = cv2.threshold(gray,gs('threshold_low'), 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    gray = 255-gray
    gray=np.round((gray/255),decimals=0)
    gray=255*gray
    gray=np.uint8(gray)
    return gray
def shift_2(img,t_invert):
    # gets the best shift base on matrix center of gravity
    # and places in a 28 by 28 pixel field
    import cv2
    from digit_capture import get_global_settings as gs
    fss=gs('final_side_size')
    import numpy as np
    t_paste=np.zeros((fss,fss),dtype='int')
    r,c=img.shape
    x,y=getBestShift(img)
    cv2.imwrite('bad_image.jpg',img)
    #print(str(y)+" "+str(x)+" "+str(r)+" "+str(c))
    if((r+y)>fss):
        import time;
        ts = time.time()
        ts = time.ctime(ts)
        y=fss-r
        f = open("digit_capture_log.txt", "a")
        f.write("had to modify center y "+str(ts)+" \n")
        f.close()
    if((c+x)>fss):
        import time;
        ts = time.time()
        ts = time.ctime(ts)
        x=fss-c
        f = open("digit_capture_log.txt", "a")
        f.write("had to modify center x "+str(ts)+"\n")
        f.close()    
    t_paste[y:(y+r),x:x+c]=img[0:r,0:c]
    if(t_invert==1):
        t_paste=255-t_paste
    return t_paste
    
def shift(img,sx,sy):
    import numpy as np
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
def set_in_bounding_square(t_image,t_side,i):
    # takes in a grayscale image and returns the same
    # image within its bounding square
    import numpy as np
    import cv2
    from digit_capture import get_global_settings as gs
    fss=gs('final_side_size')
    t_image=t_image.astype('uint8')
    t_image = cv2.resize(t_image,(fss,fss))
    #print("bounding "+str(type(t_image[0,0]))+"   "+str(i))
    #(thresh, t_image) = cv2.threshold(t_image, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #if(t_invert==0):        
    #    t_image = 255-t_image
    t_image = 255 - t_image
    while np.sum(t_image[0]) == 0:
        t_image = t_image[1:]

    while np.sum(t_image[:,0]) == 0:
        t_image = np.delete(t_image,0,1)

    while np.sum(t_image[-1]) == 0:
        t_image = t_image[:-1]

    while np.sum(t_image[:,-1]) == 0:
        t_image = np.delete(t_image,-1,1)
    rows,cols=t_image.shape
    nrows=0
    ncols=0
    if(rows>cols):
        factor=t_side/rows
        nrows=t_side
        ncols=int(factor*cols)
    else:
        factor=t_side/cols
        ncols=t_side
        nrows=int(factor*rows)
    #print(nrows)
    #print(ncols)
    #print((t_image.shape))
    new_image=cv2.resize(t_image,(ncols,nrows))
    (thresh, new_image) = cv2.threshold(new_image,gs('threshold_low'), 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return new_image
def output_data(t_data):
    #Outputs the image to a csv file
    from digit_capture import get_global_settings as gs
    t_dest=set_file_name_prefix(0)
    fname_pfx=t_dest[0]
    dmy=len(t_data)
    import numpy as np
    import cv2
    tAry=np.asarray(t_data)
    #tAry
    final_side_size=gs('final_side_size')
    grid_count = gs('grid_count')
    t_class=gs('t_class')
    df_class=make_class_list(t_class,(grid_count*grid_count))
    df_column_names=make_col_list((final_side_size*final_side_size))
    pxl=gs('pxl')
    line_wd=gs('line_wd')
    canvas_width = grid_count*pxl
    canvas_height = grid_count*pxl
    prev_x=-999999;
    prev_y=-999999;
    x_coords=[]
    y_coords=[]
    width = canvas_width  # canvas width
    height = canvas_height # canvas height
    center = height/2
    white = (255, 255, 255) # canvas back
    t_bd1=5
    t_bd2=175
    d_1=1
    d_2=gs('d_2')
    d_3=120
    d_4=180
    wdh=1
    #print(type(t_data))
    tAry=np.asarray(t_data,dtype="uint8")
    #tAry.shape
    #tAry2_dim0=int(len(tAry)/4)
    #tAry2=np.reshape(tAry,(tAry2_dim0,4))
    #nw_image=np.reshape(tAry2[0:tAry2_dim0,0],(canvas_width,canvas_width))
    #cv2.imwrite(fname_pfx+'python_version.jpg',nw_image)
    #print(' point 3 '+str(nw_image.shape))
    #print(type(nw_image[0,0]))
    nw_image=np.reshape(tAry,(canvas_width,canvas_width))
    save(nw_image)
