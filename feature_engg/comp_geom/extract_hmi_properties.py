import astropy
from astropy.io import fits
import numpy as np
import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os, glob
import networkx as nx
import sys
import matplotlib.patches as patches

from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.sparse import csr_matrix
from scipy.stats import itemfreq
from scipy.ndimage.measurements import center_of_mass
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics.pairwise import euclidean_distances

def get_distance(pt1, pt2):
    return np.linalg.norm(pt1 - pt2);

def draw_networks(data_in, idx, epsilon):
    
    data = np.zeros(data_in.shape)
    [x_idx, y_idx] = idx;
    data[x_idx, y_idx] = 1
    conn_matrix = np.zeros((data.shape[0] * data.shape[1], data.shape[0] * data.shape[1]))
    idx = np.concatenate((x_idx.reshape((-1,1)), y_idx.reshape((-1,1))), axis=1)
   
    G = nx.DiGraph();
    
    for i in range(idx.shape[0]):
        G.add_node(i);
         
    for i in range(idx.shape[0]):
        for j in range(idx.shape[0]):
            dist = get_distance(idx[i,:], idx[j,:])
            
            conn_x = idx[i,0]*data.shape[1] + idx[i,1];
            conn_y = idx[j,0]*data.shape[1] + idx[j,1];
            
            if (dist < epsilon):
                conn_matrix[conn_x, conn_y] = 1;
                conn_matrix[conn_y, conn_x] = 1;
                G.add_edge(i, j);
             
    pos = {};
    for i in range(idx.shape[0]): 
        pos[i] = (idx[i,1], -idx[i,0]);
    
    nx.set_node_attributes(G, 'coord', pos);

    return G;


def get_clusters(data_in, idx, epsilon):
    
    data = np.zeros(data_in.shape)
    [x_idx, y_idx] = idx;
    data[x_idx, y_idx] = 1
    idx = np.concatenate((x_idx.reshape((-1,1)), y_idx.reshape((-1,1))), axis=1)
    conn_matrix = np.zeros((idx.shape[0], idx.shape[0]))

    for i in range(idx.shape[0]):
        for j in range(idx.shape[0]):
            dist = get_distance(idx[i,:], idx[j,:])
            
            if (dist < epsilon):
                conn_matrix[i, j] = 1;
                conn_matrix[i, j] = 1;

    conn_matrix = csr_matrix(conn_matrix)
    Tcsr = minimum_spanning_tree(conn_matrix)
    mst  = Tcsr.toarray()

    num_graphs, labels = connected_components(mst, directed=False)

    labels_freq = (itemfreq(labels));
    #labels_freq = labels_freq[labels_freq[:,1] != 1]
    labels_freq = labels_freq[(-labels_freq[:,1]).argsort()]

    com = [];
    
    clusters = [];
    for cluster in labels_freq:
        label, count = cluster;
        i = (np.where(labels == label))[0]
        
        #x_idx = idx / (data.shape[1]);
        #y_idx = idx % (data.shape[1]);
        
        x_idx = idx[i, 0];
        y_idx = idx[i, 1];

        cluster_img = np.zeros(data.shape);
        cluster_img[x_idx, y_idx] = data_in[x_idx, y_idx];
        clusters.append(cluster_img);
        com.append(center_of_mass(cluster_img))
    
    return [clusters, labels_freq[:,1], com];
    

def extract_hmi_properties(data, label, do_plot = False):

    th1 = 200
    th2 = 5000

    p_idx = np.where(np.logical_and((data > th1), (data < th2)))
    n_idx = np.where(np.logical_and((data < -th1), (data > -th2)))
    
    p_largest_arr = [];
    n_largest_arr = [];

    p_num_clusters_arr = [];
    n_num_clusters_arr = [];

    results = [];

    diag_distance = int(np.sqrt(data.shape[0] **2 + data.shape[1] ** 2))

    for epsilon in np.arange(1.5,1.6,1):
      [p_cluster, p_freq, p_com] =  get_clusters(data, p_idx, epsilon);
      [n_cluster, n_freq, n_com] =  get_clusters(data, n_idx, epsilon);
      
      if len(p_freq) != 0:
         p_largest_arr.append(p_freq[0]);
      else:
         p_largest_arr.append(0);

      if len(n_freq) != 0:
         n_largest_arr.append(n_freq[0]);
      else:
         n_largest_arr.append(0);

      p_num_clusters_arr.append(len(p_freq));
      n_num_clusters_arr.append(len(n_freq));
      
      clusters_img = np.zeros(data.shape);
      if len(p_cluster) != 0:  
        clusters_img += p_cluster[0];

      if len(n_cluster) != 0:  
        clusters_img += n_cluster[0];
      
      if do_plot:
          plt.figure();
          plt.matshow(clusters_img);
          plt.colorbar()
          plt.savefig('clusters_' + str(epsilon) + '.png');
          plt.close('all')

      if (len(p_cluster) * len(n_cluster)) != 0:
      
        best_p_idx = 0;
        best_n_idx = 0; 
        best_force = 0;
        best_com_dist = 0;
        best_min_dist = 0;

        p_size_max = np.sum(np.where((p_cluster[0] != 0)))
        p_idx_max = len(p_cluster);
        for p_idx_cur in range(len(p_cluster)):
            p_size = np.sum(np.where((p_cluster[p_idx_cur] != 0)));
            if (p_size < .1*p_size_max):
               p_idx_max = p_idx_cur;
               break;

        n_idx_max = len(n_cluster);
        n_size_max = np.sum(np.where((n_cluster[0] != 0)))
        for n_idx_cur in range(len(n_cluster)):
            n_size = np.sum(np.where((n_cluster[n_idx_cur] != 0)));
            if (n_size < .1*n_size_max):
               n_idx_max = n_idx_cur;
               break;
        
        all_clusters_img = np.zeros(data.shape);
        for p_idx_cur in range(p_idx_max):
          for n_idx_cur in range(n_idx_max):

            p_idx_eps = np.where((p_cluster[p_idx_cur] != 0));
            n_idx_eps = np.where((n_cluster[n_idx_cur] != 0));
            p_idx_eps = np.concatenate((p_idx_eps[0].reshape((-1,1)), p_idx_eps[1].reshape((-1,1))), axis=1);        
            n_idx_eps = np.concatenate((n_idx_eps[0].reshape((-1,1)), n_idx_eps[1].reshape((-1,1))), axis=1);        
            #hausdorff_dist = directed_hausdorff(p_idx_eps, n_idx_eps);
            min_dist = np.min(euclidean_distances(p_idx_eps, n_idx_eps));
            com_dist = get_distance(np.array(p_com[p_idx_cur]), np.array(n_com[n_idx_cur]));

            p_mag = np.sum(p_cluster[p_idx_cur]);
            n_mag = np.abs(np.sum(n_cluster[n_idx_cur]));

            if best_force < (n_mag * p_mag / (min_dist ** 2)):
                best_force = n_mag * p_mag / (min_dist ** 2);
                best_p_idx = p_idx_cur;
                best_n_idx = n_idx_cur;
                best_com_dist = com_dist;
                best_min_dist = min_dist;
          
            #print(p_idx_cur, n_idx_cur, p_mag, n_mag, min_dist, (n_mag * p_mag / (min_dist ** 2)), best_force);
        
            clusters_img = np.zeros(data.shape);
            clusters_img += p_cluster[p_idx_cur];
            clusters_img += n_cluster[n_idx_cur];

            all_clusters_img += p_cluster[p_idx_cur];
            all_clusters_img += n_cluster[n_idx_cur];
            
            if do_plot: 

                plt.figure();
                plt.matshow(clusters_img);
                plt.colorbar()
                plt.savefig('all_clusters_' + str(p_idx_cur) + '_' + str(n_idx_cur) + '.png');
         
        results.append([label, epsilon,  
                      len(p_cluster),
                      len(n_cluster),
                      np.sum(np.where((p_cluster[0] != 0))),
                      np.sum(np.where((n_cluster[0] != 0))),
                      np.sum(p_cluster[0]),
                      np.sum(n_cluster[0]),
                      np.sum(np.where((p_cluster[best_p_idx] != 0))),
                      np.sum(np.where((n_cluster[best_n_idx] != 0))),
                      np.sum(p_cluster[best_p_idx]),
                      np.sum(n_cluster[best_n_idx]),
                      np.sum(p_cluster[best_p_idx]) / np.sum(np.where((p_cluster[best_p_idx] != 0))), 
                      np.sum(n_cluster[best_n_idx]) / np.sum(np.where((n_cluster[best_n_idx] != 0))),
                      best_com_dist, best_min_dist, best_com_dist / best_min_dist, best_force]);

        print(results)
            
        clusters_img = np.zeros(data.shape);
        clusters_img += p_cluster[best_p_idx];
        clusters_img += n_cluster[best_n_idx];
            
        if do_plot: 

            plt.figure();
            plt.matshow(clusters_img);
            plt.savefig('best_clusters_' + str(p_idx_cur) + '_' + str(n_idx_cur) + '.png');

            plt.figure();
            plt.matshow(all_clusters_img);
            plt.savefig('all_clusters_' + str(p_idx_cur) + '_' + str(n_idx_cur) + '.png');

      elif len(p_cluster) == 0 and len(n_cluster) == 0:
        
        results.append([label, epsilon,  
                      len(p_cluster),
                      len(n_cluster),
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0, 
                      0,
                      diag_distance, diag_distance, 1, 0]);
        
      elif len(p_cluster) == 0:
        
        results.append([label, epsilon,  
                      len(p_cluster),
                      len(n_cluster),
                      0,
                      np.sum(np.where((n_cluster[0] != 0))),
                      0,
                      np.sum(n_cluster[0]),
                      0,
                      0,
                      0,
                      0,
                      0, 
                      0,
                      diag_distance, diag_distance, 1, 0]);
      
      elif len(n_cluster) == 0:
        
        results.append([label, epsilon,  
                      len(p_cluster),
                      len(n_cluster),
                      np.sum(np.where((p_cluster[0] != 0))),
                      0,
                      np.sum(p_cluster[0]),
                      0,
                      0,
                      0,
                      0,
                      0,
                      0, 
                      0,
                      diag_distance, diag_distance, 1, 0]);

    
    if do_plot:

        plt.figure();
        plt.plot(p_largest_arr, label = 'positive')
        plt.plot(n_largest_arr, label = 'negative')
        plt.title('Largest $\epsilon$-component Size')
        plt.xlabel('Epsilon');
        plt.ylabel('Largest Component Size');
        plt.legend();
        plt.savefig('largest_size_' + label + '.png');

        plt.figure();
        plt.plot(p_num_clusters_arr, label = 'positive')
        plt.plot(n_num_clusters_arr, label = 'negative')
        plt.title('Number of $\epsilon$-components')
        plt.xlabel('Epsilon');
        plt.ylabel('# Components');
        plt.legend();
        plt.savefig('num_clusters_' + label + '.png');

    return results;

def draw():
    
    label = sys.argv[1];
    epsilon = float(sys.argv[2]);

    hdu = fits.open('./data/hmi.sharp_720s.7115.' + label + '_TAI.magnetogram.fits'); 
    hdu[1].verify('fix')
    data = hdu[1].data[::4,::4] 

    th1 = 200
    th2 = 5000
    
    pos_thr_str = str(th1) + '_' + str(th2) ;
    neg_thr_str = str(-th2) + '_' + str(-th1);

    p_idx = np.where(np.logical_and((data > th1), (data < th2)))
    n_idx = np.where(np.logical_and((data < -th1), (data > -th2)))

    G_pos = draw_networks(data, p_idx, epsilon);
    G_neg = draw_networks(data, n_idx, epsilon);
    
    title = '$\epsilon$-connected components ($\epsilon$ = %.1f)' % epsilon;
    name = 'scc_' + label + '_' + pos_thr_str + '_{}.png'.format(epsilon);

    fig, ax = plt.subplots(1, figsize=(10,6));
    ax.plot([0], [0], '.', color = 'blue', label = 'positive')
    ax.plot([0], [0], '.', color = 'red', label = 'negative')
    #plt.title(title)

    nx.draw_networkx_nodes(G_pos, node_color = 'b', edge_color = 'b', node_size = 1, pos=nx.get_node_attributes(G_pos, 'coord'), ax = ax);
    nx.draw_networkx_edges(G_pos, pos=nx.get_node_attributes(G_pos, 'coord'), edge_color = 'b',ax = ax, alpha = 0.3);
    nx.draw_networkx_nodes(G_neg, node_color = 'r', edge_color = 'r', node_size = 1, pos=nx.get_node_attributes(G_neg, 'coord'), ax = ax);
    nx.draw_networkx_edges(G_neg, pos=nx.get_node_attributes(G_neg, 'coord'), edge_color = 'r',ax = ax, alpha = 0.3);
    plt.tick_params(axis='both',       # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    left=False,      # ticks along the bottom edge are off
                    right=False,         # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False, 
                    labeltop=False,
                    labelright=False) # labels along the bottom edge are off

    #plt.legend();
    plt.savefig(name);
    
    plt.figure();
    plt.matshow(data);
    plt.colorbar();
    plt.savefig('data.png');

if __name__ == '__main__':

    label = sys.argv[1];
    hdu = fits.open('/srv/data/varad/data/fits/sharp_7115/hmi.sharp_cea_720s.7115.' + label + '_TAI.Br.fits'); 
    hdu[1].verify('fix')
    data = hdu[1].data[::4,::4] 
    #draw()
    results = extract_hmi_properties(data, label, do_plot=True);
    #np.savetxt('results.txt', results, fmt='%.3f');
