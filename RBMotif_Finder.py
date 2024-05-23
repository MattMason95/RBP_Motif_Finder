class RBMotif_finder:
    '''
        RBmotif_finder is a function for analysing submitted fasta sequences for the presence of RNA-binding factor motif sites.
        
    '''
    import pandas as pd
    import numpy as np
    import itertools
    from itertools import islice
    import random
    
    def __init__(self, fasta_path, motif_path, background=True, threshold=0.8, measurement='hamming'):
        self.fasta_path = fasta_path
        self.motif_path = motif_path
        self.background = background
        self.threshold = threshold
        self.measurement = measurement   

        valid_measurements = ['hamming']
        if self.measurement not in valid_measurements:
            raise Exception(f"Invalid parameter. {measurement} was passed to the measurement parameter. Only {valid_measurements} are permitted.")
        
        if isinstance(self.threshold, float):
            pass
        else:
            raise Exception(f"Invalid data type. Only integers can be passed to threshold parameter.")       
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def load_fasta(self):
        ## Extract FASTA data from the path provided
        fasta = pd.read_csv(self.fasta_path,sep='\t',header=None)
        labels = fasta.iloc[::2][0].to_list()
        sequences = fasta.iloc[1::2][0].to_list()

        if len(labels) != len(sequences):
            raise Exception('Mismatch between number of extracted FASTA labels and sequences.')
        
        return labels, sequences
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def load_motifs(self):
        ## Extract motif data from the path provided
        motifs = pd.read_csv(self.motif_path,header=None)
        motif_labels = motifs.iloc[:,0]
        motif_seqs = motifs.iloc[:,1]
        
        if len(motif_labels) != len(motif_seqs):
            raise Exception('Mismatch between number of extracted motif labels and sequences.')
        return motif_labels, motif_seqs
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def hamming_distance(self,chain1,chain2):
        self.chain1 = chain1
        self.chain2 = chain2

        return sum(c1 != c2 for c1, c2 in zip(chain1,chain2))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def window(self,sequence,n):
        '''
        Taken from Itertools documentation: 
        Returns a sliding window (of width n) over data from the iterable.
        s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
        '''
        it = iter(sequence)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result    
        for elem in it:
            result = result[1:] + (elem,)
            yield result
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def background_model(self,sequence):
        '''
        In order to evaluate the statistical significance of a detected consensus motif, the consensus scores must be compared against a background model.
        The background model generates a randomly shuffled version of the input sequence.
        
        '''
        random.seed(42)

        scrambled  = ''.join(random.sample(sequence,len(sequence)))
        
        return scrambled
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def compute_distances(self):
        labels,sequences = self.load_fasta()
        motif_labels,motif_seqs = self.load_motifs()

        print(f'Loaded {len(labels)} FASTA sequences.')
        print(f'Scanning with {len(motif_labels)} RNA binding motifs.')
        print('---------------------------------------------------------------')
        print(f'Calculating distance with {self.measurement} metric.')
        # print(sequences)

        if self.measurement == 'hamming':
            output = {}

            for idx, (motif_lab, motif_seq) in enumerate(zip(motif_labels,motif_seqs)):
                motif_length = len(motif_seq)

                print(f'Loaded {len(labels)} FASTA sequences.')
                print(f'Computing Z-scores for {motif_seq} motif enrichment.')
                print('---------------------------------------------------------------')

                temp_list = np.empty((len(sequences),300-motif_length))
                background_list = np.empty((len(sequences),300-motif_length))
                
                for idx, (fasta_label, fasta_seq) in enumerate(zip(labels,sequences)):
                    sequence_slices = [(motif_length-self.hamming_distance(x,motif_seq))/motif_length for x in self.window(fasta_seq, motif_length)]
                    scrambled_slices = [(motif_length-self.hamming_distance(x,motif_seq))/motif_length for x in self.window(self.background_model(fasta_seq), motif_length)]
                    try:
                        temp_list[idx,:] = sequence_slices
                        background_list[idx,:] = scrambled_slices
                    except: 
                        temp_list[idx,:] = sequence_slices[:-1]
                        background_list[idx,:] = scrambled_slices[:-1]

                total = temp_list.sum(axis=0)
                # mean = np.mean(background_list.sum(axis=0))
                # std = np.std(background_list.sum(axis=0))
                mean = np.mean(temp_list.sum(axis=0))
                std = np.std(temp_list.sum(axis=0))
                
                
                Z_scores = np.array([(x-mean)/std for x in total])
                output[f'{motif_lab}:{motif_seq}'] = Z_scores
           
        return output
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def plotting(self,save=False,plot_type=['heatmap','kdeplot'],save_name=None):
        if plot_type == 'heatmap':
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_theme(context='poster',style='white',font_scale=1)

            outputs = pd.DataFrame(self.compute_distances()).T

            # blend_cmap = sns.blend_palette(["#4D8C9A","#3EDCAA"],200, as_cmap=True)
            blend_cmap = sns.blend_palette(["#30668A",'#E2C046',"#C14428"],200, as_cmap=True)
            blend_cmap.set_under((1.0,1.0,1.0,0.0))

            fig, ax = plt.subplots(1,1,figsize=(15,10))

            fig = sns.heatmap(outputs,vmin=1,vmax=11,cmap=blend_cmap,zorder=1,cbar_kws={"shrink": 0.5,'aspect':10,"ticks":[1,4,7,10]})
                        
            ax.set_xticks(ticks=[0,147,294],labels=['-150','0','+150'],rotation=0,fontweight='bold')
            ax.hlines(y=list(range(1,len(outputs.index))),xmin=0,xmax=294,linestyle='dashed',lw=1,color='0.2',clip_on=False)
            ax.axhline(y=len(outputs.index),xmin=0,xmax=0.999,lw=3,color='0.2',clip_on=False)
            ax.axvline(x=0,ymin=-0.02,ymax=0,color='0.2',clip_on=False)
            ax.axvline(x=147,ymin=-0.02,ymax=0,color='0.2',clip_on=False)
            ax.axvline(x=294,ymin=-0.02,ymax=0,color='0.2',clip_on=False)
            ax.axvline(x=147,ymin=0,ymax=1.05,lw=2,ls=':',color='0.2',clip_on=False,zorder=0)
            ax.set_xlabel('Genomic Position from Intron-Exon Junction (bp)',fontweight='bold')
            
            if save:
                fig.figure.savefig(f'Figures//{save_name}_heatmap.png',dpi=250,bbox_inches='tight')
            
            return outputs   
