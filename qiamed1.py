##################################################################################################################
##**Project: Quantum Intelligent Agent for Medical Diagnosis (QIAMed)
####
##** Quantum Deep Q-Learning (QDQL) simulation for cancer diagnosis:
##-- Without using any quantum computing libraries
##-- Simulate quantum behavior by probability amplitudes and stochastic action sampling
##-- Include both learning and multiple-class predictions
##
##** Comments:
##-- As the model is only at the beginning of training, it is possible to have equal probabilities on predictions. 
##-- when there is a high uncertainty of membership of the symptoms in the defined classes.
##################################################################################################################

import numpy as np
import random


# Simulated patient data (symptom vector + correct diagnosis index)
# --- Data: 5 cancer types, 19 symptoms (binary vector input) ---
# Liste des 19 symptômes

symptom_list = [
    "persistent cough", "chest pain", "shortness of breath", "weight loss",
    "lump in breast", "breast pain", "nipple discharge", "skin dimpling",
    "frequent urination", "weak urine stream", "pelvic discomfort",
    "blood in stool", "abdominal pain", "unexplained weight loss", "fatigue",
    "new mole", "changing mole", "itchy skin", "bleeding mole"
]

# 5 classes of cancer types
cancer_types = ["Lung", "Breast", "Prostate", "Colorectal", "Skin"]

# Cancer classes (actions)
env = [
    # Lung Cancer (class 0)
    ([1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 0),
    ([1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 0),
    # Breast Cancer (class 1)
    ([0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0], 1),
    ([0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0], 1),
    # Prostate Cancer (class 2)
    ([0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0], 2),
    ([0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0], 2),
    # Colorectal Cancer (class 3)
    ([0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0], 3),
    ([0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0], 3),
    # Skin Cancer (class 4)
    ([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1], 4),
    ([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1], 4)
]

# --- Parameters ---

num_symptoms = 19  # Simulated symptom input size
num_actions = len(cancer_types)
alpha = 0.5  # learning rate
gamma = 0.95 # discount factor
epochs = 300

class QuantumDeepQLearning:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon  # Exploration rate
    
        # Initialize Q-table (simulate quantum-like state space with 2^features states)
        # Q-table: state (2⁴=16) x actions (5 cancer types)
    
        self.Q = np.zeros((2**num_symptoms, num_actions))  # state-action table       

    # Convert binary symptom list to state index
    def state_index(self, symptoms):
        return int("".join(map(str, symptoms)), 2)

    # Simulated quantum-inspired softmax policy
    def quantum_sim_policy(self, state_q_values):
        exp_vals = np.exp(state_q_values - np.max(state_q_values))
        probs = exp_vals / np.sum(exp_vals)
        action = np.random.choice(len(probs), p=probs)
        return action
        
    # --- Prediction with softmax probabilities ---
    def predict_with_probabilities(self, symptoms):
        state = self.state_index(symptoms)
        q_vals = self.Q[state]
        exp_vals = np.exp(q_vals - np.max(q_vals))
        probs = exp_vals / np.sum(exp_vals)
        return probs
    
    def prediction_result(self, test_input):
        self.probs1 = self.predict_with_probabilities(test_input)
        return self.probs1

agent = QuantumDeepQLearning()

print("Quantum Deep Q-Learning (QDQL) Simulation for Cancer Diagnosis")
print("\n")

# --- Training loop ---
print("Simple Training:")

for epoch in range(epochs):
    symptoms, correct = random.choice(env)
    state = agent.state_index(symptoms)
    action = agent.quantum_sim_policy(agent.Q[state])
    reward = 1 if action == correct else -1
    agent.Q[state][action] += alpha * (reward + gamma * np.max(agent.Q[state]) - agent.Q[state][action])
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Associated symptoms with {cancer_types[action]}: Reward = {reward}")     

print("\n")
# Test examples

test_input1 = [0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0]
liste1 = []

test_input2 = [1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
liste2 = []

test_input3 = [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
liste3 = []
##
for i in range(num_symptoms):
    if test_input1[i] == 1:
        liste1.append(symptom_list[i])
        
for i in range(num_symptoms):
    if test_input2[i] == 1:
        liste2.append(symptom_list[i])
        
for i in range(num_symptoms):
    if test_input3[i] == 1:
        liste3.append(symptom_list[i])
##

print("Input value 1 (Finding symptoms):", ", ".join(liste1))

print("Diagnostic predictions on all classes:")

test_input = test_input1
agent.prediction_result(test_input)
for i, prob in enumerate(agent.probs1):
        print(f"{cancer_types[i]}: {round(prob * 100, 2)}%")   

if np.argmax(agent.probs1) == 0:
    print("There is uncertainty in the diagnosis. Please investigate further with tests.")
else:
   print(f"Suggested predicted value corresponds to : {cancer_types[np.argmax(agent.probs1)]}")

print("\n")
##
print("Input value 2 (Finding symptoms):", ", ".join(liste2))
print("Diagnostic predictions on all classes:")

test_input = test_input2
agent.prediction_result(test_input)
for i, prob in enumerate(agent.probs1):
        print(f"{cancer_types[i]}: {round(prob * 100, 2)}%")   

if np.argmax(agent.probs1) == 0:
    print("There is uncertainty in the diagnosis. Please investigate further with tests.")
else:
   print(f"Suggested predicted value corresponds to : {cancer_types[np.argmax(agent.probs1)]}")

print("\n")
##
print("Input value 3 (Finding symptoms):", ", ".join(liste3))
print("Diagnostic predictions on all classes:")

test_input = test_input3
agent.prediction_result(test_input)
for i, prob in enumerate(agent.probs1):
        print(f"{cancer_types[i]}: {round(prob * 100, 2)}%")   

if np.argmax(agent.probs1) == 0:
    print("There is uncertainty in the diagnosis. Please investigate further with tests.")
else:
   print(f"Suggested predicted value corresponds to : {cancer_types[np.argmax(agent.probs1)]}")

