# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 09:43:06 2021

@author: Ehsan Saghapour
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from PIL import Image


# st.markdown('Streamlit is **_really_ cool**.')
st.title('Create GeneTerrain')
# st.text('by Ehsan Saghapour, PhD., Jake Chen, PhD.')
# st.text('GeneTerrain consist of 2 parts including  ')
btn6= st.checkbox('Information about GeneTerrain')


if btn6:
    st.write("A network visualization technique using scattered data interpolation and surface rendering, based upon a foundation layout of a scalar field. [1]")

    st.write("[1] You, Q., Fang, S. and Chen, J.Y., 2010. Gene terrain: Visual exploration of differential gene expression profiles organized in native biomolecular interaction networks. Information Visualization, 9(1), pp.1-12.")
btn5= st.checkbox('Information about Gene Expression and Netrwork File')

@st.cache
def cr_gt(result,sigma,res):  # creat of Gene terrain 
    
    x_ = np.linspace(0, 10, res)
    y_ = np.linspace(0, 10, res)
    X, Y = np.meshgrid(x_, y_)
    
    gaussian=np.zeros((res,res))
    
    pi=3.14
    
    for i in range(len(result)):
        
        x=10*result.iloc[i,0]
        y=10*(1-result.iloc[i,1])
        amp1=result.iloc[i,3]
    
        gaussian1 =(1/np.sqrt(2*pi*sigma))*np.exp(-(((X-x)/sigma)**2+((Y-y)/sigma)**2)) 
        gaussian =amp1* gaussian1 + gaussian
        # gaussian = np.flipud(gaussian)
    gray=(gaussian-np.min(gaussian))/(np.max(gaussian)-np.min(gaussian))
    
    return gray

# @st.cache
def plot_interaction(network,width, height ): 
    
    G=nx.Graph()
    G.add_weighted_edges_from(network)
    # G = nx.random_geometric_graph(200, 0.125)
    pos=nx.spring_layout(G)
    # labels = x1.dtype.names
    # x1 = x1.tolist()    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]] 
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=.5, color='blue'),
        hoverinfo='none',
        mode='lines')
    
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        textposition='top center',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            colorscale='Hot',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(list(G.nodes)[node].astype(str))
    
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig1 = go.Figure(data=[edge_trace, node_trace],
                  layout=go.Layout(

                    showlegend=False,
                    hovermode='closest',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig1.update_layout(
    autosize=False,
    width=width,
    height=height,

    )
    
    return fig1


def plot_sub_interaction(Layout1,network2,consider_gene,res ): 
        
        sub_network = networkk2.loc[networkk2['Gene X'] == consider_gene]
        print(sub_network)
        sub_network1 = networkk2.loc[networkk2['Gene Y'] == consider_gene]
        print(sub_network1)
        sub_network1.columns=['Gene Y', 'Gene X', 'Sigma'] 
        sub_network=pd.concat([sub_network,sub_network1], axis=0)
        print(sub_network)

        sub_net=sub_network['Gene Y'].tolist()
        print(len(sub_net))
        sub_network = np.core.records.fromarrays(sub_network.values.T,
                                          names='col1, col2, col3',
                                          formats = 'S5, S5, f8')
        
            
        G=nx.Graph()
        G.add_weighted_edges_from(sub_network)
        # G = nx.random_geometric_graph(200, 0.125)
        # pos=nx.spring_layout(G)
        # labels = x1.dtype.names
        # x1 = x1.tolist()    
        edge_x = []
        edge_y = []
        idxx=np.where(Layout1['name']==consider_gene)[0]
        s1= np.array(Layout1.iloc[idxx,1:3])
        
        for name in sub_net:
            idxx=np.where(Layout1['name']==name)[0]
            s = np.array(Layout1.iloc[idxx,1:3])
             
            edge_x.append(np.round(res*(s1[0][0]))-1)
            edge_x.append(np.round(res*s[0][0])-1)
            edge_x.append(None)
            edge_y.append(np.round(res*(1-s1[0][1])))
            edge_y.append(np.round(res*(1-s[0][1])))
            edge_y.append(None)
        
        edge_trace=go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='blue'),
            # hoverinfo='none',
            mode='lines',
            name='lines ('+ consider_gene +')'
                    )
        node_x = []
        node_y = []
        # sub_net.append(consider_gene)
        # sub_net1=consider_gene..append(sub_net)
        
        idxx=np.where(Layout1['name']==consider_gene)[0]
        s= np.array(Layout1.iloc[idxx,1:3])
        node_x.append(np.round(res*(s[0][0]))-1)
        node_y.append(np.round(res*(1-s[0][1])))
        
        
        for node in sub_net:
            idxx=np.where(Layout1['name']==node)[0]
            s = np.array(Layout1.iloc[idxx,1:3])
            node_x.append(np.round(res*s[0][0])-1)
            node_y.append(np.round(res*(1-s[0][1])))
        
      
        node_trace=go.Scatter(
            x=res*node_x, y=res*node_y,
            # text=result.index
            text=sub_net,
            textposition='top center',
            mode='text+markers',
            name='Point ('+ consider_gene +')'
        )

        #     mode='lines+text+markers',
        node_adjacencies = []
        node_adjacencies.append(len(sub_net))
        node_text = []
        
        node_text.append(consider_gene)
        for node in sub_net:
            node_adjacencies.append(1)
            node_text.append(node)
        
        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

        return edge_trace , node_trace


def specific_gt(networkk2,sub_gt,result,sigma):   # Sub Gene terrains

# Gene_expr=pd.read_csv('Expression.txt',sep='\t')
# Sample_ID=Gene_expr['sampleID']

    networkk = pd.read_csv('Network.txt',sep='\t',header=None)
    networkk2=networkk
    networkk2.columns =['Gene X', 'Gene Y', 'Sigma'] 

    x_ = np.linspace(0, 10, res)
    y_ = np.linspace(0, 10, res)
    X, Y = np.meshgrid(x_, y_)

    sub_network = networkk2.loc[networkk2['Gene X'] == sub_gt]
    # print(sub_network)
    sub_network1 = networkk2.loc[networkk2['Gene Y'] == sub_gt]
    # print(sub_network1)
    sub_network1.columns=['Gene Y', 'Gene X', 'Sigma'] 
    sub_network=pd.concat([sub_network,sub_network1], axis=0)

    gaussian=np.zeros((res,res))
    
    
    sub_networkk=sub_network['Gene Y'].tolist()
    sub_networkk.append(sub_gt)
    
    
    for name in sub_networkk:
        
        # if name is result.index:
        # sigma= .1
        # print(np.where(result.index==name)[0])
        if len(np.where(result.index==name)[0])>0 :
            x,y=result.iloc[np.where(result.index==name)[0],0:2].values[0]
            x=10*(x)
            y=10*(1-y)
            amp1=result.iloc[np.where(result.index==name)[0],3].values[0]
    
            
            # gaussian1 =(1/np.sqrt(2*pi*sigma))*np.exp(-(((X-x)/sigma)**2+((Y-y)/sigma)**2)) 
            gaussian1 =np.exp(-(((X-x)/sigma)**2+((Y-y)/sigma)**2)) 
    
            gaussian =amp1* gaussian1 + gaussian
        # gaussian = np.flipud(gaussian)
    gray=(gaussian-np.min(gaussian))/(np.max(gaussian)-np.min(gaussian))
    fig = px.imshow(1-gray,color_continuous_scale='spectral',width=1000, height=1000)
    # gray=(gaussian-np.min(gaussian))/(np.max(gaussian)-np.min(gaussian))
            

    return fig


def Info_data(Gene_expr,networkk2):
    st.header( "Information about Gene Expression File:")
    st.write( "The Number of Samples and Genes are:" , np.shape(Gene_expr)  , ", respectively. " )
    # st.write( "The number of Genes is:" , np.shape(Gene_expr)  )
    if np.shape(Gene_expr)[0] > 10:
        st.write('The 5 First Rows of Gene Expression File (For Datasets that are more than 5 Samples):', Gene_expr.head())
    else: 
        st.write('Gene Expression File:',Gene_expr)
    
    if networkk2 is not None:
    # networkk2=networkk = pd.read_csv('Network.txt',sep='\t',header=None)
        networkk2.columns =['Gene X', 'Gene Y', 'Sigma'] 
        
        
        ss=networkk2['Gene X'].value_counts()
        
        ss1=networkk2['Gene Y'].value_counts()
        ss1=pd.DataFrame(ss1)
        ss=pd.DataFrame(ss)
        ss=ss.reset_index()
        ss1=ss1.reset_index()
        ss=ss.merge(ss1, how='inner', on='index')
        ss['Total']=ss['Gene X']+ss['Gene Y']
        ss=ss.sort_values(by=['Total'],ascending=False)
        ss.reset_index(drop=True, inplace=True) 
        
        st.header( "Information about Network File:")
        st.write( "The Number of Gene-Gene Interactions:" , len(networkk2) )
        st.write('The 5 First Rows of Network File :', networkk2.head())
        dd=ss.iloc[:,[0,3]].head(10)
        dd.columns=['Name of Gene', '# Interactions' ]
        st.write('The 10 First Genes that have more interaction with another Genes:', dd)

def normliaze_data(data):
    for col in Gene_expr:
        data[col]=(data[col] - np.min(data[col])) / (np.max(data[col]) - np.min(data[col]))
    return data


# @st.cache
def file_selector():
    file = st.sidebar.file_uploader("Choose a Gene_expression file", type="csv")
    if file is not None:
      # Gene_expr=pd.read_csv(file,sep='\t')
      Gene_expr=pd.read_csv(file,sep=',')
      return Gene_expr
    else:
      st.sidebar.text("Choose a Gene_expression file")
Gene_expr=file_selector()


# @st.cache
def file_selector1():
    file = st.sidebar.file_uploader("Choose a Layout file", type="txt")
    if file is not None:
      Layout1=pd.read_csv(file,sep='\t',header=None,index_col=None)
      return Layout1
    else:
      st.sidebar.text("Choose a Layout file")
Layout_=file_selector1()

btn2= st.sidebar.checkbox('W/Gene names')
btn3= st.sidebar.checkbox('Save GeneTerrain')



# # @st.cache
def file_selector2():
    file = st.sidebar.file_uploader("Choose a Network file", type="txt")
    if file is not None:
      # networkk = np.genfromtxt(file, delimiter="\t", names=None, usecols=[0,1,2],dtype=['S5','S5','f8'])
      networkk = pd.read_csv(file,sep='\t',header=None)

      return networkk
    else:
      st.sidebar.text("Choose a Network file")
networkk=file_selector2()




btn=False
if Gene_expr is not None:  
        Sample_ID = Gene_expr['sampleID']
        
        if btn5:  
            if networkk is not None:
                Info_data(Gene_expr,networkk)
            
        chosen_ID = st.selectbox(
            "Choose Sample_ID", Sample_ID
        )
        Gene_expr=Gene_expr.set_index('sampleID')
        # Gene_expr= normliaze_data(Gene_expr)
        Gene_expr=Gene_expr.T
        
        Gene_expr1=Gene_expr[chosen_ID]
        if networkk is  None:
            btn = st.button("Create_GeneTerrain")


res=st.sidebar.slider('Resulotion of image',100, 1000, 200,100)
if btn3:
    res1=st.slider('Change the Resulotion of image to save it',100, 1000, res,50)
# Gene_expr=pd.read_csv('Mesenchymal_profileExpression.txt',sep='\t',header=None)
# Layout_=pd.read_csv('Layout_equalWeight.txt',sep='\t',header=None,index_col=None)
# print(uploaded_file)
# unique_name=np.unique(networkk)
from io import BytesIO
import base64

# import cv2
def get_image_download_link(img,res1):
 
    new_p = Image.fromarray(np.uint8(img * 255))

    # new_p = cv2.cvtColor(np.float32(new_p),cv2.COLOR_GRAY2RGB)
    # new_p = cv2.merge([new_p,new_p,new_p])
    # if new_p.mode == "F":        
    new_p = new_p.convert('RGB') 
    
    new_p = new_p.resize((res1, res1))
    buffered = BytesIO()
    
    new_p.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    name_ID=chosen_ID +'.jpeg'

    # href = f'<a href="data:file/jpg;base64,{img_str}">Download result</a>'
    href = '<a href="data:file/jpg;base64,{}" download={} >Download (Click Here)</a>'.format(img_str,name_ID)
    
    # href ='<a href="{}" rel="noopener noreferrer" target="_blank">{}</a>'.format(img_str,name_ID)
    st.markdown(href, unsafe_allow_html=True)
    
if networkk is not None:
    
    networkk2=networkk
    networkk2.columns =['Gene X', 'Gene Y', 'Sigma'] 
    target_options =  np.unique(networkk2.iloc[:,0:2])
    # name_unique = st.selectbox(
    # "Choose Gene", target_options)
    # print(name_unique)   
    selected_indices = st.multiselect('Choose Gene(s):', target_options)
    
    # st.write('With selectiong more sigma value, you will see more interaticon among genes')
    sigma=st.slider('Sigma (With selectiong more sigma value, you will see more interaticon among genes)',.0, 1.0, .1,.05)
    
    btn = st.button("Create_GeneTerrain")
    # selected_rows = data.loc[selected_indices]
    # st.write( selected_indices)

fig2 = make_subplots(rows=2, cols=1)
if btn: 
    # Gene_expr=Gene_expr.set_index(0) 
    Layout_=Layout_.set_index(0) 
    
    result = pd.concat([Layout_, Gene_expr1], axis=1)
    # result.rename(columns={'X','Y','Sigma', 'Ampuitled'})
    result=pd.DataFrame(result)  
    result=result.rename(columns={1: "X",2: "Y", 3: "Sigma"})

    result=result.dropna()

    if btn2:
        
            # gaussian = np.flipud(gaussian)
        gray = cr_gt(result,sigma,res)
        fig = px.imshow(gray,color_continuous_scale='spectral',width=1000, height=1000)
        fig.add_trace(go.Scatter(
            x=np.round(res*result.iloc[:,0]),
            y=np.round(res*(1-result.iloc[:,1])),
            text=result.index,
            textposition='top center',
            mode='text+markers',
            name='Point'
            )
            
        )
    else:   
        Layout1=pd.read_csv('Layout_equalWeight.txt',sep='\t',header=None,index_col=None)

        Layout1.columns =['name','X', 'Y', 'Sigma']

        networkk = pd.read_csv('Network.txt',sep='\t',header=None)
        networkk2=networkk
        networkk2.columns =['Gene X', 'Gene Y', 'Sigma'] 
 
        print(selected_indices)
        if len(selected_indices)>0:
            fig = specific_gt(networkk2,selected_indices[0],result,sigma)
            node_edge, node_trace= plot_sub_interaction(Layout1,networkk2,selected_indices[0],res)
            fig.add_trace(node_trace)
            fig.add_trace(node_edge)
        else:
            gray = cr_gt(result,sigma,res)
            fig = px.imshow(gray,color_continuous_scale='spectral',width=1000, height=1000)
            

        fig.update_layout(coloraxis_showscale=False)
    if btn3 :
        fig.write_html(chosen_ID +".html")
        import matplotlib.cm
        import matplotlib.pyplot as plt

        cm_hot = matplotlib.cm.get_cmap('jet')
        get_image_download_link(cm_hot(1-gray),res1)
        
        plt.plot(gray)
        # figss = plt.figure()
        figss = plt.figure()
        #plt.figure(figsize=(3.841, 7.195), dpi=100)
        ax = figss.add_subplot(111)
        gray=ax.matshow(gray, cmap="jet")
        print(gray)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(chosen_ID +".jpeg", bbox_inches = 'tight',
        pad_inches = 0)
        
 
        # Show matrix in two dimensions
    st.plotly_chart(fig)
    

if btn is True and networkk is not None: 
    networkk1 = np.core.records.fromarrays(networkk.values.T,
                            names='Gene X,Gene Y, Sigma',
                            formats = 'S5, S5, f8')
    st.plotly_chart(plot_interaction(networkk1,800, 800 ))

st.sidebar.text("##################################")


