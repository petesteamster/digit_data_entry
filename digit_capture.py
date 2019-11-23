def insert_mods_to_js_file(t_js_file,the_dict):
    from digit_capture import set_file_name_prefix
    from digit_capture import read_pickle,write_pickle
    # this will read in a js file. Inputs will be
    # the file name.
    # It will retrieve the .pk file
    # that has the dictionary that holds the mods to
    # be inserted. It will return the file with the replaced
    # values. This function enables rudumentry communication
    # from python to javascript
    fname_pfx=set_file_name_prefix(0)[0]
    #pk_file=t_js_file[0,-3]+'.pk'
    #the_dict=read_pickle(fname_pfx+pk_file)
    #file1 = open(fname_pfx+t_js_file,"r+")
    file1 = open(t_js_file,"r+")  
    the_js_data=file1.read()
    for old,new in the_dict.items():
        the_js_data=the_js_data.replace(old,str(new))
    return the_js_data
#insert_mods_to_js_file(t_js_file,t_dict)
def cross_ref_replace_dict(t_file_name):
    import pandas as pd
    from digit_capture import set_file_name_prefix
    from digit_capture import get_global_settings as gs
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
                 "?width?",
                 "?height?"]
    test_values = [0,0,0,0,0,0,0,0,0,0,t_full_side_size,t_full_side_size] 
    # using dictionary comprehension 
    # to convert lists to dictionary 
    res = {test_keys[i]: test_values[i] for i in range(len(test_keys))} 
    fname_pfx=set_file_name_prefix(0)[0]
    df=pd.read_csv(fname_pfx+t_file_name,header=None)
    t_df=df.iloc[:,0:2]
    t_list=['class','replace_dict']
    t_df.columns = t_list
    dd=t_df.groupby('class').count().to_dict()
    #print(dd['replace_dict'])
    for k, v in dd['replace_dict'].items():
        #print(k,dd['replace_dict'][k])
        res[k]=dd['replace_dict'][k]
    res['?grid_count?']=get_dict()['grid_count']
    res['?pxl?']=get_dict()['pxl']
    res['?line_wd?']=get_dict()['line_wd']
    return res
def get_dict():
    #dictionary holds the values of global variables.
    gbl={'grid_count':5,
         'line_wd':5,
         'pxl':50,
         'd_2':50,
         'final_side_size':28,
         't_class':'?class02?'
        }
    return gbl
def get_global_settings(t_var):
        #used to look up the values of global variables 
    gbl=get_dict()
    rtn=gbl[t_var]
    return rtn
def mount_drive():
    if(not in_colab()):
        return
    from google.colab import drive
    drive.mount('/content/drive')
def in_colab():
    import sys
    IN_COLAB = 'google.colab' in sys.modules
    return IN_COLAB
def set_file_name_prefix(colab):
    # if colab=1, set up structure for colab.
    # if colab=0, set up structure for local.
    # if colab=2, set up structure when in /content/drive  
    import sys
    import os
    t_prefix=""
    t_path=""
    if(in_colab()):
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
            t_rs=cv2.resize(t_d,(final_side_size,final_side_size))
            im_list.append(t_d)
            im_reshaped[t_index,]=t_rs
            t_index=t_index+1
            st_2=st_2+t_dim
        st_1=st_1+t_dim
        st_2=0
    return (im_list,im_reshaped)                
def save(t_canvas_image):
    #function used to save imaages to a csc file.
    #It 
    
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
    for i in range(0,(grid_count*grid_count)):
        new_out=np.reshape(new_data[i,:],(28,28))
        t_ind=str(i).zfill(5)
        cv2.imwrite(fname_pfx+'digit'+t_ind+'.jpg',new_out)
    print(len(t_digits))
def write_digit_data(t_data,file_name):
    import pandas as pd
    t_data.to_csv(file_name,header=False,mode='a',index=False)
def make_class_list(t_class,class_count):
    import pandas as pd
    lst=[t_class]*class_count
    df=pd.DataFrame(lst)
    return df                    
def do_pca(t_data):
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
    print(pca.n_components_)
    train_data = pca.transform(train_data)
    return train_data
def write_pickle(file_name,obj_name):
    import pickle as pk
    print(file_name)
    #with open(file_name,'wb') as f:
    #    pk.dump(obj_name,f)
    pk.dump( obj_name, open( file_name, "wb" ) )
def read_pickle(file_name):
    import pickle as pk
    rtn=0
    with open(file_name,'rb') as f:
        rtn=pk.load(f)
    return rtn    
def getBestShift(img):
    import numpy as np
    import scipy.ndimage as ndi
    cy,cx = ndi.measurements.center_of_mass(255-img)
    shiftx = np.round(28/2.0-cx).astype(int)
    shifty = np.round(28/2.0-cy).astype(int)
    return shiftx,shifty

def make_col_list(t_count):
    rtn=[]
    rtn.append('t_class')
    for i in range(0,t_count):
        rtn.append('col_'+str(i).zfill(4))
    return rtn

def reshape_mnist(tData):
    import numpy as np
    entries=tData.shape[0]
    new_format=np.zeros((entries,(28*28)),dtype='float')
    for i in range(0,entries):
        (thresh, gray) = cv2.threshold(tData[i,:,:], 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        new_format[i,:]=np.reshape(gray,(1,28*28))
    return new_format    
def reformat_mnist(tData,t_invert):
    import numpy as np
    entries=tData.shape[0]
    new_format=np.zeros((entries,(28*28)),dtype='uint8')
    for i in range(0,entries):
        t_image=set_in_bounding_square(tData[i,:,:],20,i)
        shifted=shift_2(t_image,t_invert)
        shifted=np.reshape(shifted,(28*28,))
        new_format[i,:]=shifted
    return new_format    
def load_and_threshold_file(t_file):    
    import cv2
    import numpy as np
    gray1=cv2.imread(t_file,0)
    gray=cv2.imread(t_file,0)
    gray = cv2.resize(gray,(28, 28))
    (thresh, gray) = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    gray = 255-gray
    gray=np.round((gray/255),decimals=0)
    gray=255*gray
    gray=np.uint8(gray)
    return gray
def shift_2(img,t_invert):
    import numpy as np
    t_paste=np.zeros((28,28),dtype='int')
    r,c=img.shape
    x,y=getBestShift(img)
    t_paste[y:(y+r),x:x+c]=img
    if(t_invert==1):
        t_paste=255-t_paste
    return t_paste
    
def shift(img,sx,sy):
    import numpy as np
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
def set_in_bounding_square(t_image,t_side,i):
    import numpy as np
    import cv2
    t_image=t_image.astype('uint8')
    t_image = cv2.resize(t_image,(28, 28))
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
    (thresh, new_image) = cv2.threshold(new_image, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return new_image
def output_data(t_data):
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
    center = height//2
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
    tAry2_dim0=int(len(tAry)/4)
    tAry2=np.reshape(tAry,(tAry2_dim0,4))
    nw_image=np.reshape(tAry2[0:tAry2_dim0,0],(canvas_width,canvas_width))
    cv2.imwrite(fname_pfx+'python_version.jpg',nw_image)
    print(type(nw_image[0,0]))
    save(nw_image)
