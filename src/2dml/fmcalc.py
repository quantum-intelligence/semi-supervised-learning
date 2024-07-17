import numpy as np
#import urllib2
import matplotlib.pyplot as plt
import re
import pandas as pd
#from urlparse import urljoin
#from bs4 import BeautifulSoup
import pickle
import os.path
from ase.spacegroup import Spacegroup


#  
# read the data from the URL and print it
#

def openurl(url):
    # open a connection to a URL using urllib2
    webUrl = urllib2.urlopen(url)
  
    # get the result code and print it
    # print "result code: " + str(webUrl.getcode()) 
  
    # read the data from the URL and print it
    data = webUrl.read()
    webUrl.close()
    return data

def href_filter(tag):
    return tag.has_attr('href') #and not tag.has_attr('id')

def index_only(tag):
    return tag.has_attr('href') and not tag.has_attr('class')

def pull_urls(html_main):  
    """
    Pull urls from main html page and attach href to the base url address
    """
    base = 'http://webbdcrista1.ehu.es/magndata/index.php?show_db=1'
    soup = BeautifulSoup(html_main, 'html.parser')
    href_list = []
    all_html = soup.find_all(href=re.compile("index="), src=False,attrs={'class':'blue'})
    for url in all_html:
        href_leaf = url.get('href')
        href_full = urljoin(base,href_leaf)
        href_list.append(href_full)
    return href_list


def get_abc_index(tb):
    """get index for 'Lattice parameters of the magnetic unit cell:' in html code """
    bvalues = []
    for i in np.arange(30):
        bval = tb[i].string
        bvalues.append(bval)
    abc_index = bvalues.index('Lattice parameters of the magnetic unit cell:')
    return abc_index




def get_m(data):
    """
    gets mx, my, mz, m_tot from parsed url using openurl()
    Keeps track on m_i from more than one atomic species
    """
    tomatobasil = BeautifulSoup(data,'lxml')
    table = tomatobasil.find(id="only_mag") 
    telem = table.next_element
    rows = telem.findChildren('tr')
    numAtom = len(rows)-1
    atomlist = []
    iter = 0
    for row in rows:
        iter=iter+1
        cells = row.findChildren('td')
        rowinfo = []
        for cell in cells:
            rowinfo.append(cell.string)
        if iter != 1: #ignore header
            atomlist.append(rowinfo)
    # clean up strings in atomlist:
    for j, item in enumerate(atomlist):
        for i, elem in enumerate(atomlist[j]):
            if elem==None:
                atomlist[j][i] = ""
            else:
                atomlist[j][i] = re.sub("[\(\[].*?[\)\]]", "", elem.string) 
    return atomlist  # this will be nested list with one element if only one term is present

def get_spacegroup_index(tb):
    """get index for 'Parent space group' in html code """
    bvalues = []
    for i in np.arange(10):
        bval = tb[i].string
        bvalues.append(bval)
    sg_index = bvalues.index('Parent space group')
    return sg_index


def get_spacegroup(tomatobasil,sg):
    """ gets spacegroup info from html codes. extracts it from a tag and url link"""
    sg_val =  tomatobasil.find_all('b')[sg]
    sg_str = (sg_val.find_next_sibling("a")).next_element
    sg_href = tomatobasil.find_all(href=re.compile('gen&gnum'))
    sg_url = (sg_href[0].get('href'))
    gnum = re.sub(r'http.+gnum=', '', sg_url) 
    sg_list = sg_str.split()
    return [sg_list, gnum]




def get_Tc(tomatobasil):
    """
    - get index for 'Parent space group' in html code 
    - gets spacegroup info from html codes. extracts it from a tag and url link
    """
    #get index value:
    tb = tomatobasil.find_all('b')
    bvalues = []
    for i in np.arange(30):
        bval = tb[i].string
        bvalues.append(bval)
    try:
        tc_index = bvalues.index('Transition Temperature: ')
    except ValueError:
        tcurie = 0
    else:
        #"Do something with variable b"
        tc_val =  tomatobasil.find_all('b')[tc_index]
        tc_str = tc_val.next_sibling
        tc_str = re.sub("[\(\[].*?[\)\]]", "", tc_str)
        tc_list = tc_str.split()
        tcurie = (tc_list[0])
        tcurie = str(tcurie)
        #remove hyphens
        tcurie = tcurie.replace('-',' ')
        tcurie = tcurie.split()
        #print tcurie
        if len(tcurie)>1:
            #print 'here'
            #print type(tcurie)
            tcurie = np.asfarray(tcurie)
            #print type(tcurie)
            tcurie = np.mean(tcurie)
    return tcurie

#aa = get_Tc(tomatobasil)
#print 'aa', aa



def get_mag_unitcell(tomatobasil,abc):
    """ gets lattice parateres for magnetic unit cell from html codes"""
    #tb_soup = tomatobasil.find_all('b')
    #abc = get_abc_index(tb_soup)
    ucell =  tomatobasil.find_all('b')[abc]
    ucell_str = (ucell.find_next_sibling("br")).next_element
    ucell_str = re.sub("[\(\[].*?[\)\]]", "", ucell_str) 
    ucell_list = ucell_str.split()
    return ucell_list





def get_name(tomatobasil):
    #tomatobasil = BeautifulSoup(data,'html.parser')
    #tb_soup = tomatobasil.find_all('b')
    #print 'tb', tomatobasil.find_all('h2')
    fname_id =  tomatobasil.find_all('h2')[1].text  #take second element and transform to text/string
    fname_id = fname_id.split()
    fname = fname_id[0]
    id = fname_id[1]
    id = re.sub("[(#)]", "", id) 
    return [fname, id]



def getinfo(url):
    data = openurl(url)
    tomatobasil = BeautifulSoup(data,'html.parser')
    tb_soup = tomatobasil.find_all('b')
    fname =  get_name(tomatobasil)  #take second element and transform to text/string
    abc = get_abc_index(tb_soup)
    ucell_list = get_mag_unitcell(tomatobasil,abc)
    #
    #get spacegroup:
    sg = get_spacegroup_index(tb_soup)
    sg_val = get_spacegroup(tomatobasil,sg)
    #print 'string',(ucell_str),type(ucell_str)
    #print ucell_str.split()
    mdata = get_m(data)
    tcurie = get_Tc(tomatobasil)
    return fname, ucell_list, mdata, sg_val, tcurie



def getallinfo(magdataurls,max):
    infolist = []
    i = 0
    for url in magdataurls:
        if i < max:
            i = i +1
            fname, ucell_str, mdata, sg_val, tcurie = getinfo(url)
            info = [fname[0],fname[1], ucell_str, mdata, sg_val, tcurie]
            infolist.append(info)

    return infolist




## DATA EXTRACTION FUNCTIONS (from DATAFRAME)

def extract_magmom(df):
    """ Extracts magnetic moment info from dataframe 
    and outputs a list of numpyarrays
    """
    mag_mom = df['mag_mom']
    atomlist = []
    mnetlist = []
    for atom in mag_mom:
        marray = []
        for m_i in atom:
            mom = m_i[-4:] #get lasts for entries, m_x, M_y, m_z and m_tot
            mom = np.asarray(mom)
            marray.append(mom)
        marray  = np.asfarray(marray)
        msub = marray[:,:3]
        m_sum = np.sum(msub, axis=0)
        m_net = np.sqrt(np.sum(m_sum**2.0))
        mnetlist.append(m_net)
        atomlist.append(marray)
    mnetlist = np.asfarray(mnetlist)
    atomlist = np.asarray(atomlist)
    return atomlist, mnetlist



def extract_s_group(df):
    """reads symmetry group information from dataframe"""
    sg_elem = df["s_group"]
    #sg_array = np.zeros((df.shape[0],1))
    sg_array = []
    counter = 0
    for i, elem in enumerate(sg_elem):
        #if len(elem) == 2:
        #    sg = elem[1]
        #else:
        #    sg = 0 #no digit given for spacegroup code
        #    print elem[0]
        #    sg_string = (str(elem[0]))
        #    sg_symbol = Spacegroup(sg_string)
        #    counter = counter + 1
        sg = elem[-1:]
        sg = np.float(sg[0])
        sg_array.append(sg)
    sg_array = np.asarray(sg_array)
    return sg_array
    


def calc_vol(lat_array,angle_array):
    a = lat_array[:,0]
    b = lat_array[:,1]
    c = lat_array[:,2]
    alpha = angle_array[:,0]*np.pi/180.
    beta = angle_array[:,1]*np.pi/180.
    gamma = angle_array[:,2]*np.pi/180.
    a_c = a*c
    vol = a*b*c*np.sqrt(1.0+2.0*np.cos(alpha)*np.cos(beta)*np.cos(gamma)-(np.cos(alpha))**2-(np.cos(beta))**2-(np.cos(gamma))**2)
    log_vol = np.log(vol)
    return log_vol, vol

def calc_mtot(atomlist):
    """
    takes the m_tot for the first atomic species only
    """
    m_tot=[]
    for item in atomlist:
        first_atom = item[0][-1] #
        m_tot.append(first_atom)
    return m_tot


def extract_lat(df):
    lat_elem = df["mag_lat_param"]
    lat_array = np.zeros((df.shape[0],3))
    angle_array = np.zeros((df.shape[0],3))
    for i, elem in enumerate(lat_elem):
        lat = elem[:6]
        lat = np.asarray(lat)
        lat_array[i] = lat[:3]
        angle_array[i] = lat[-3:]
    return lat_array, angle_array
