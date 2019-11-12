def restartkernel():
    from IPython.display import display_html
    display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)
def separate_digits(t_image,t_dim,grid_count):
    import numpy as np
    import cv2
    global final_side_size 
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
def save1(t_canvas_image):
    #function used to save imaages to a csc file.
    #It 
    
    import pandas as pd
    import cv2
    import numpy as np
    #filename = "user_input.jpg"
    #output_image.save(filename)
    #read in image
    global d_2
    global master
    global w
    global grid_count
    global df_class
    global df_column_names
    out_array = np.zeros(((grid_count*grid_count),(d_2*d_2)),dtype='uint8')
    #t_image=img = cv2.imread('user_input.jpg',0)
    (t_digits,t_reformat_dg)=separate_digits(t_canvas_image,d_2,grid_count)
    #print('save '+str(t_reformat_dg.shape))
    new_data=reformat_mnist(t_reformat_dg,0)
    df_dta=pd.DataFrame(data=new_data)
    df_dta=pd.concat([df_class,df_dta],axis=1)
    write_digit_data(df_dta,'digit_data.csv')
    for i in range(0,(grid_count*grid_count)):
        new_out=np.reshape(new_data[i,:],(28,28))
        t_ind=str(i).zfill(5)
        cv2.imwrite('digit'+t_ind+'.jpg',new_out)
    #new_data=reformat_mnist(t_reformat_dg,0)
    print(len(t_digits))
    ##master.quit()
    #master.destroy()
    #master.quit()
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
def read_pickle(file_name,obj_name):
    import pickle as pk
    rtn=0
    with open(file_name,'rb') as f:
        rtn=pk.load(obj_name,f)
    return rtn    
def getBestShift(img):
    import numpy as np
    import scipy.ndimage as ndi
    cy,cx = ndi.measurements.center_of_mass(255-img)
    #print(str(cy)+" cy")
    #print(str(cx)+ "cx")
    #rows,cols = img.shape
    shiftx = np.round(28/2.0-cx).astype(int)
    shifty = np.round(28/2.0-cy).astype(int)
    #print(str(shiftx)+ " shift x")
    #print(str(shifty)+ " shift y")
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
    #np.zeros((28,28),dtype='int')
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
    print(type(gray))
    gray = cv2.resize(gray,(28, 28))
    (thresh, gray) = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    gray = 255-gray
    #(thresh, gray) = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    gray=np.round((gray/255),decimals=0)
    gray=255*gray
    gray=np.uint8(gray)
    return gray
def shift_2(img,t_invert):
    import numpy as np
    t_paste=np.zeros((28,28),dtype='int')
    r,c=img.shape
    x,y=getBestShift(img)
    #print(str(r) + " " + 'r')
    #print(str(c) + " " + 'c')
    #print('shape img '+ str(img.shape)) 
    t_paste[y:(y+r),x:x+c]=img
    if(t_invert==1):
        t_paste=255-t_paste
    return t_paste
    
def shift(img,sx,sy):
    import numpy as np
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    #print((M.shape))
    #print((img.shape))
    #shifted = cv2.warpAffine(img,M,(cols,rows))
    #return shifted  
def set_in_bounding_square(t_image,t_side,i):
    import numpy as np
    import cv2
    t_image=t_image.astype('uint8')
    t_image = cv2.resize(t_image,(28, 28))
    print("bounding "+str(type(t_image[0,0]))+"   "+str(i))
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
