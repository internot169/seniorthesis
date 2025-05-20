import spacy
import networkx as nx
import matplotlib.pyplot as plt
import re

class KnowledgeGraphGenerator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.graph = nx.DiGraph()
        
    def preprocess_text(self, text):
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append((ent.text, ent.label_))
        return entities
    
    def extract_relationships(self, text):
        doc = self.nlp(text)
        relationships = []
        
        for sent in doc.sents:
            sent_entities = []
            for ent in doc.ents:
                if ent.start >= sent.start and ent.end <= sent.end:
                    sent_entities.append(ent)
            
            for i in range(len(sent_entities) - 1):
                ent1 = sent_entities[i]
                ent2 = sent_entities[i + 1]
                start = ent1.end
                end = ent2.start
                words = []
                for token in doc[start:end]:
                    if not token.is_punct:
                        words.append(token.text)
                
                if words:
                    relation = ' '.join(words)
                else:
                    relation = 'related_to'
                
                relationships.append((ent1.text, relation, ent2.text))
            
            for token in sent:
                if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                    subject = token.text
                    verb = token.head.text
                    for child in token.head.children:
                        if child.dep_ in ["dobj", "pobj", "attr"]:
                            obj = child.text
                            relationships.append((subject, verb, obj))
        
        return relationships
    
    def build_graph(self, text):
        cleaned_text = self.preprocess_text(text)
        entities = self.extract_entities(cleaned_text)
        relationships = self.extract_relationships(cleaned_text)
        
        for entity, label in entities:
            self.graph.add_node(entity, type=label)
        
        for subject, relation, obj in relationships:
            if subject in self.graph and obj in self.graph:
                self.graph.add_edge(subject, obj, relation=relation)
    
    def visualize_graph(self, output_file="knowledge_graph.png"):
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', node_size=2000, alpha=0.6)
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', arrows=True, arrowsize=20)
        nx.draw_networkx_labels(self.graph, pos, font_size=10)
        
        edge_labels = nx.get_edge_attributes(self.graph, 'relation')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        
        plt.axis('off')
        plt.savefig(output_file)
        plt.close()