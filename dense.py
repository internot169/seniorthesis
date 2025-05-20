import spacy
import networkx as nx
import matplotlib.pyplot as plt
import re

class RawGraphGenerator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.graph = nx.Graph()
        
    def preprocess_text(self, text):
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def build_graph(self, text):
        cleaned_text = self.preprocess_text(text)
        doc = self.nlp(cleaned_text)
        
        for sent in doc.sents:
            words = [token.text.lower() for token in sent if not token.is_punct and not token.is_space]
            
            for i in range(len(words)):
                for j in range(i + 1, len(words)):
                    self.graph.add_edge(words[i], words[j])
    
    def visualize_graph(self, output_file="raw_graph.png"):
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', node_size=2000, alpha=0.6)
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray')
        nx.draw_networkx_labels(self.graph, pos, font_size=10)
        
        plt.axis('off')
        plt.savefig(output_file)
        plt.close()