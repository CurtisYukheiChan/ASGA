#V0.11
from group_logic import create_groups  
from flask import Flask, request, render_template, jsonify, send_file, send_from_directory, session
import pandas as pd
import logging
import uuid
import statistics
from queue import Queue
import traceback
import io
from collections import Counter, defaultdict
import os
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import os
import random
import tempfile
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, BaseDocTemplate, Frame, PageTemplate
from reportlab.platypus import Image
import matplotlib.pyplot as plt
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

import numpy as np
import time
from pylatex import Document, Section, Math
import math
from threading import Thread


from openpyxl import load_workbook

import csv
import copy
from copy import deepcopy
import pandas as pd

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
task_queue = Queue()
results={}
app.config['UPLOAD_FOLDER'] = 'uploads'
# VARIABLES TO STORE
groups = None
UPDATE_INTERVAL = 50  # milliseconds
participants=[]
temp_storage = {}  # server-side store
app.secret_key = os.urandom(24)


#Matrix solver

# matrix solver

# for rendering properly and DIP aware in monitor
import ctypes

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # For Windows 8.1 or later
except:
    try:
        ctypes.windll.user32.SetProcessDPIAware()  # For Windows 7
    except:
        pass



#  uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#loading variables, CSV file

#genetic algo
def compute_fitness(groups, skill_cap, min_max_skill_A_prior,min_max_skill_B_prior,min_max_skill_C_prior, min_skill_enforcement, ESL_weight, gender_weight,whitelist_weight,blacklist_weight, motivation_weight, teamwork_weight,basis_motivation,basis_IE,basis_TWS):
    motivation_std = np.mean([
        np.std([p['motivation'] for p in group])
        for group in groups
        if len(group) >= 2
    ]) if any(len(group) >= 2 for group in groups) else 0
    motivation_std = np.nan_to_num(motivation_std, nan=0.0)


    IE_std = np.mean([
        np.std([p['IE'] for p in group])
        for group in groups
        if len(group) >= 2
    ]) if any(len(group) >= 2 for group in groups) else 0

    IE_std = np.nan_to_num(IE_std, nan=0.0)  # replace NaN with 0.0

    teamwork_std = np.std([
        np.mean([p['teamwork'] for p in group])
        for group in groups
        if len(group) >= 2
    ]) if any(len(group) >= 2 for group in groups) else 0

    teamwork_std = np.nan_to_num(teamwork_std, nan=0.0)  # replace NaN with 0.0


    skill_penalty = 0
    diversity_penalty_sum = sum(
        diversity_penalty(group, gender_weight, ESL_weight)
        for group in groups if len(group) >= 4
    ) * 10
    diversity_penalty_sum = np.nan_to_num(diversity_penalty_sum, nan=0.0)  # replace NaN with 0.0


    whitelist_bonus = 0
    blacklist_penalty = 0

    for group in groups:
        if len(group) == 0:
            continue

        # Skills checking
        max_skills = {
            'A': max(p['skill_A'] for p in group),
            'B': max(p['skill_B'] for p in group),
            'C': max(p['skill_C'] for p in group)
        }
        skill_sum = sum(max_skills.values())
        if skill_sum < skill_cap:
            skill_penalty += (skill_cap - skill_sum) * 1


        if max_skills['A'] < min_skill_enforcement * min_max_skill_A_prior:
            skill_penalty += 1

        if max_skills['B'] < min_skill_enforcement * min_max_skill_B_prior:
            skill_penalty += 1

        if max_skills['C'] < min_skill_enforcement * min_max_skill_C_prior:
            skill_penalty += 1

        if math.isnan(skill_penalty):
            skill_penalty = 0


        #  whitelist and blacklist scoring
        total_whitelist, total_blacklist = count_all_matches(group)

        # whitelist matches
        whitelist_bonus += total_whitelist * whitelist_weight
        if math.isnan(whitelist_bonus):
            whitelist_bonus = 0

        #  blacklist matches
        blacklist_penalty += total_blacklist * blacklist_weight

        if math.isnan(blacklist_penalty):
            blacklist_penalty = 0



    # Fitness aims to maximize motivation and IE_std but minimize penalties
    return -((motivation_std * motivation_weight)/basis_motivation + skill_penalty*20 + diversity_penalty_sum + blacklist_penalty - IE_std/basis_IE - whitelist_bonus + (teamwork_std*teamwork_weight)/basis_TWS )


def generate_initial_population(participants, num_groups, group_size, population_size):
    population = []
    for _ in range(population_size):
        shuffled = random.sample(participants, len(participants))
        groups = [shuffled[i * group_size:(i + 1) * group_size] for i in range(num_groups)]
        population.append(groups)
    return population


def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(list(zip(population, fitnesses)), k)
    selected.sort(key=lambda x: x[1], reverse=True)
    return deepcopy(selected[0][0])


def crossover(parent1, parent2, group_size):
    # combine participants from both parents
    all_participants = [p for group in parent1 for p in group]

    # shuffle and split again while preserving group sizes
    random.shuffle(all_participants)
    group_sizes = [len(group) for group in parent1]
    new_groups = []
    index = 0
    for size in group_sizes:
        new_groups.append(all_participants[index:index + size])
        index += size
    return new_groups



def mutate(groups, mutation_rate=0.05):
    for _ in range(int(mutation_rate * sum(len(g) for g in groups))):
        g1, g2 = random.sample(range(len(groups)), 2)
        if groups[g1] and groups[g2]:
            i1 = random.randint(0, len(groups[g1]) - 1)
            i2 = random.randint(0, len(groups[g2]) - 1)
            groups[g1][i1], groups[g2][i2] = groups[g2][i2], groups[g1][i1]
    return groups


def genetic_algorithm(participants, num_groups, group_size, skill_cap, min_max_skill_A_prior,min_max_skill_B_prior,min_max_skill_C_prior, min_skill_enforcement,
                      population_size, max_generations, groups_sorted_C, ESL_weight, gender_weight,whitelist_weight,blacklist_weight, motivation_weight, teamwork_weight,basis_motivation,basis_IE,basis_TWS):
    population = [copy.deepcopy(groups_sorted_C) for _ in range(population_size)]
    print(f"motivation weight: {motivation_weight}")

    for generation in range(max_generations):
        fitnesses = [
            compute_fitness(groups, skill_cap, min_max_skill_A_prior,min_max_skill_B_prior,min_max_skill_C_prior, min_skill_enforcement, ESL_weight, gender_weight,whitelist_weight,blacklist_weight,motivation_weight,teamwork_weight,basis_motivation,basis_IE,basis_TWS)
            for groups in population]

        new_population = []
        for _ in range(population_size):
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover(parent1, parent2, group_size)
            child = mutate(child)
            child = repair_duplicates(child, participants)
            new_population.append(child)

        population = new_population
        best_fitness = max(fitnesses)
        print(f"Generation {generation}: Best Fitness = {best_fitness:.4f}")
        #if generation % 5 == 0:
            #lone_gender, lone_esl = count_lone_groups(population[np.argmax(fitnesses)])
            #print(f"Gen {generation}: Lone Gender: {lone_gender:.2f}%, Lone ESL: {lone_esl:.2f}%")

        # early stopping if convergence is detected
        if generation > 5 and all(np.isclose(f, best_fitness, atol=1e-6) for f in fitnesses[-5:]):
            break

    best_index = np.argmax(fitnesses)
    return population[best_index]


# sub algorithm

def make_serializable(groups):
    serializable_groups = []
    for group in groups:
        serializable_group = []
        for p in group:
            # Strip unnecessary fields or convert NumPy to native types
            serializable_group.append({
                'id': p['id'],
                'name': p.get('name', ''),
                'skill_A': float(p['skill_A']),
                'skill_B': float(p['skill_B']),
                'skill_C': float(p['skill_C']),
                'motivation': float(p['motivation']),
                'IE': float(p['IE']),
                'email': p['email'],
                'gender': p.get('gender',''),
                'ESL': p.get('ESL', ''),
                'whitelist': p.get('whitelist', ''),
                'blacklist': p.get('blacklist', ''),
                'personality': float(P['personality']),



            })
        serializable_groups.append(serializable_group)
    return serializable_groups


def repair_duplicates(groups, all_participants):
    participant_ids = {id(p) for p in all_participants}
    current_ids = [id(p) for group in groups for p in group]

    id_counts = Counter(current_ids)

    # Find duplicates and missing participants
    duplicates = [p for p in all_participants if current_ids.count(id(p)) > 1]
    missing = [p for p in all_participants if id(p) not in current_ids]

    # Flatten groups to fill in new ones
    flat_groups = [p for group in groups for p in group]

    # Remove duplicates
    seen = set()
    cleaned = []
    for p in flat_groups:
        pid = id(p)
        if pid not in seen:
            cleaned.append(p)
            seen.add(pid)

    # Add missing participants
    cleaned += missing

    # Re-split into groups of the original size
    group_sizes = [len(group) for group in groups]
    new_groups = []
    index = 0
    for size in group_sizes:
        new_groups.append(cleaned[index:index + size])
        index += size
    return new_groups


def count_lone_groups(groups):
    print(f"Type of groups: {type(groups)}")
    print(f"First group type: {type(groups[0]) if groups else 'empty groups'}")
    print(f"First participant dict in first group: {groups[0][0] if groups and groups[0] else 'empty'}")
    lone_gender_count = 0
    lone_esl_count = 0
    lone_gender_group_ids = []
    lone_esl_group_ids = []

    for idx, group in enumerate(groups):
        genders = [p['gender'] for p in group if str(p['gender']).lower() != 'other']
        esls = [p['ESL'] for p in group]

        # Lone gender check
        if len(set(genders)) > 1:
            counts = [genders.count(label) for label in set(genders)]
            if min(counts) == 1:
                lone_gender_count += 1
                lone_gender_group_ids.append(idx + 1)

        # Lone ESL check
        if len(set(esls)) > 1:
            counts = [esls.count(label) for label in set(esls)]
            if min(counts) == 1:
                lone_esl_count += 1
                lone_esl_group_ids.append(idx + 1)

    total_groups = len(groups)
    percent_lone_gender = (lone_gender_count / total_groups) * 100
    percent_lone_esl = (lone_esl_count / total_groups) * 100

    return percent_lone_gender, percent_lone_esl, lone_gender_group_ids, lone_esl_group_ids




def diversity_penalty(group, ESL_weight, gender_weight):
    genders = [p['gender'] for p in group if str(p['gender']).lower() != 'other']
    esls = [p['ESL'] for p in group]
    penalty = 0

    # Lone gender
    if len(set(genders)) > 1:
        counts = [genders.count(g) for g in set(genders)]
        if 1 in counts:
            penalty += 1 * gender_weight  # There's a lone gender

    # Lone ESL
    if len(set(esls)) > 1:
        counts = [esls.count(e) for e in set(esls)]
        if 1 in counts:
            penalty += 1 * ESL_weight  # There's a lone ESL status

    return penalty


def compute_max_skill(groups):
    max_skills_A = []
    max_skills_B = []
    max_skills_C = []
    for group in groups:
        max_skill_A = max(p['skill_A'] for p in group)
        max_skill_B = max(p['skill_B'] for p in group)
        max_skill_C = max(p['skill_C'] for p in group)

        max_skills_A.append(max_skill_A)
        max_skills_B.append(max_skill_B)
        max_skills_C.append(max_skill_C)

    min_max_skill_A = min(max_skills_A)
    min_max_skill_B = min(max_skills_B)
    min_max_skill_C = min(max_skills_C)
    return min_max_skill_A + min_max_skill_B + min_max_skill_C


def ways_with_leftovers(num_participants, group_size):
    n=num_participants
    base_group_count = n // group_size
    leftover = n % group_size

    # build group sizes list, assigning leftover students one by one
    group_sizes = [group_size] * base_group_count
    for i in range(leftover):
        group_sizes[i % len(group_sizes)] += 1

    # Calculate denominator: product of factorials of group sizes (order inside groups)
    denom_groups = 1
    for size in group_sizes:
        denom_groups *= math.factorial(size)

    # Number of groups
    k = len(group_sizes)

    # Calculate number of ways:
    # n! / ( (group_size1)! * (group_size2)! * ... * k! )
    ways = math.factorial(n) // (denom_groups * math.factorial(k))
    print(f"{ways:.3e}")

    return ways

def count_all_matches(group):
    total_whitelist_matches = 0
    total_blacklist_matches = 0

    # Build a set of valid participant IDs
    ids_in_group = {p['id'] for p in group}

    for participant in group:
        participant_id = participant.get('id')
        whitelist = participant.get('whitelist', [])
        blacklist = participant.get('blacklist', [])

        # Count how many people in their whitelist/blacklist are in the group (excluding themselves)
        total_whitelist_matches += sum(1 for pid in whitelist if pid in ids_in_group and pid != participant_id)
        total_blacklist_matches += sum(1 for pid in blacklist if pid in ids_in_group and pid != participant_id)

    return total_whitelist_matches, total_blacklist_matches





# main algorithm

def start_allocation(participants,
            academic_weight,
            normalise_mode,
            ESL_weight,
            skill_weight,
            min_skill,
            gender_weight,
            selected_algorithm,
            max_iterations,
            max_generations,
            population_size,
            group_size,
            num_participants,
            rounding_mode,
            show_names,
            show_email,
            show_report,
            show_gender,
            show_ESL,
            whitelist_weight,
            blacklist_weight,
            motivation_weight,
            teamwork_weight         ):
    global elapsed_time
    min_skill_enforcement=min_skill




    print(f"Selected algorithm: {selected_algorithm}")

    if rounding_mode == True:
        print("true")
    else:
        print("false")

    try:
        start_time = time.time()  # Start timer

        MAX_ITERATIONS = max_iterations

        print("checckpoint 2.5")

        try:
            print(f"num participants = {num_participants}, group_size = {group_size}")
            MIN_ITERATIONS = math.floor(
                math.log10(math.log10(ways_with_leftovers(num_participants, group_size))) * 10000 * 1.5
            )

            print("MIN_ITERATIONS:", MIN_ITERATIONS)


        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for participants and group size.")
        except Exception as e:
            messagebox.showerror("Error2", str(e))
        print("checkpoint 3")
        # Iteration calculation based on number of combination
        MIN_ITERATIONS = math.floor(
            math.log10(math.log10(ways_with_leftovers(num_participants, group_size))) * 10000 * 1.5)

        if MIN_ITERATIONS < MAX_ITERATIONS:
            MAX_ITERATIONS = MIN_ITERATIONS
        else:
            pass

        # rounding number of groups up or down
        if rounding_mode == True:
            num_groups = num_participants // group_size
            group_list = list()
        else:
            num_groups = math.ceil(num_participants / group_size)
            group_list = list()

        # Sort participants by skill A level
        participants = sorted(participants, key=lambda p: p['skill_A'], reverse=True)
        #print("Check point 1")

        while len(group_list) < num_groups:
            group_list.append([])  # Create empty groups as needed
        for i in range(0, num_groups):
            group_list[i].append(participants[i])

        # sort existing group by skill B level, then assign remaining participants by skill B level to weakest skill B team first
        groups_sorted_B = sorted(group_list, key=lambda g: max(p['skill_B'] for p in g))
        removal_list = []
        for i in range(0, num_groups):
            removal_list.append(participants[i]['id'])

        participants = [p for p in participants if p['id'] not in removal_list]

        participants = sorted(participants, key=lambda p: p['skill_B'], reverse=True)

        if len(participants) >= num_groups:
            for i in range(0, num_groups):
                groups_sorted_B[i].append(participants[i])
            # sort existing group by skill C level
            groups_sorted_C = sorted(group_list, key=lambda g: max(p['skill_C'] for p in g))
            # remove assigned participants
            removal_list = []
            for i in range(0, num_groups):
                removal_list.append(participants[i]['id'])

            participants = [p for p in participants if p['id'] not in removal_list]

            participants = sorted(participants, key=lambda p: p['skill_C'], reverse=True)
        else:
            groups_sorted_C = groups_sorted_B

        if group_size >= 3 and len(participants) >= num_groups:
            for i in range(0, num_groups):
                groups_sorted_C[i].append(participants[i])
            removal_list = []
            for i in range(0, num_groups):
                removal_list.append(participants[i]['id'])

            participants = [p for p in participants if p['id'] not in removal_list]
        else:
            pass
        print("checkpoint 4")
        # Assign participants round robin
        #print("Check point 2")
        for i, participant in enumerate(participants):
            group_index = i % len(groups_sorted_C)  # cycle through group indices
            groups_sorted_C[group_index].append(participant)

        # Initialize lists to store the max values per group
        groups = groups_sorted_C
        max_skills_A = []
        max_skills_B = []
        max_skills_C = []
        print("checkpoint 4.2")
        for group in groups:
            max_skill_A = max(p['skill_A'] for p in group)
            max_skill_B = max(p['skill_B'] for p in group)
            max_skill_C = max(p['skill_C'] for p in group)

            max_skills_A.append(max_skill_A)
            max_skills_B.append(max_skill_B)
            max_skills_C.append(max_skill_C)
            if not group:
                continue
        print("checkpoint 4.5")
        min_max_skill_A_prior = min(max_skills_A)
        min_max_skill_B_prior = min(max_skills_B)
        min_max_skill_C_prior = min(max_skills_C)


        print("checkpoint 4.7")
        NUM_GROUPS = num_participants // group_size
        skill_cap = np.floor(compute_max_skill(groups) * skill_weight)
        print("checkpoint 5")
        def compute_range(groups):
            avg_scores = [np.mean([p['motivation'] for p in group]) for group in groups]
            return max(avg_scores) - min(avg_scores)

        print(compute_max_skill(groups))

        # Local Search
        current_range = compute_max_skill(groups)

        checkpoints = set(int(MAX_ITERATIONS * i / 20) for i in range(1, 21))  # 5%, 10%, ..., 100%
        prev_avg_motivation_std = None
        no_change_count = 0
        motivation_std_records = []
        #print("Check point 3")

        average_motivation_std_basis=[]
        for group in groups:
            if len(group) > 2:
                group_motivation = [p['motivation'] for p in group]
                average_motivation_std_basis.append(np.std(group_motivation))
            else:
                average_motivation_std_basis.append(1)
        basis_motivation=np.mean(average_motivation_std_basis)


        basis_TWS = np.std([
            np.mean([p['teamwork'] for p in group])
            for group in groups
            if len(group) >= 2
        ]) if any(len(group) >= 2 for group in groups) else 1

        basis_IE = np.mean([
            np.std([p['IE'] for p in group])
            for group in groups
            if len(group) >= 2
        ]) if any(len(group) >= 2 for group in groups) else 1


        if math.isnan(basis_motivation) or basis_motivation == 0:
            basis_motivation=1
        if math.isnan(basis_TWS) or basis_TWS == 0:
            basis_TWS=1
        if math.isnan(basis_IE) or basis_IE == 0:
            basis_IE=1

        if selected_algorithm in ("local search", "combined"):
            #print("Check point 4")

            for iteration in range(MAX_ITERATIONS):
                # Select two random participants from different groups
                g1, g2 = random.sample(range(NUM_GROUPS), 2)
                i1 = random.randint(0, len(groups[g1]) - 1)
                i2 = random.randint(0, len(groups[g2]) - 1)

                if selected_algorithm == "local search":
                    motivation_std_before = np.std([p['motivation'] for p in groups[g1]]) + \
                                            np.std([p['motivation'] for p in groups[g2]])
                    diversity_penalty_before = diversity_penalty(groups[g1], ESL_weight, gender_weight) + diversity_penalty(
                        groups[g2], ESL_weight, gender_weight)
                    before_whitelist_total1, before_blacklist_total1 = count_all_matches(groups[g1])
                    before_whitelist_total2, before_blacklist_total2 = count_all_matches(groups[g2])

                    before_teamwork_std = np.std([
                        np.mean([p['teamwork'] for p in group])
                        for group in groups
                        if len(group) >= 2
                    ]) if any(len(group) >= 2 for group in groups) else 0


                if selected_algorithm == "combined":
                    #print("checkpoint 6")
                    original_g1 = groups[g1][:]
                    original_g2 = groups[g2][:]
                    fitness_before1 = compute_fitness([original_g1], skill_cap,min_max_skill_A_prior,min_max_skill_B_prior, min_max_skill_C_prior, min_skill_enforcement,
                                                     ESL_weight, gender_weight, whitelist_weight, blacklist_weight,
                                                     motivation_weight,teamwork_weight,basis_motivation,basis_IE,basis_TWS)
                    fitness_before2 = compute_fitness([original_g2], skill_cap, min_max_skill_A_prior,min_max_skill_B_prior ,min_max_skill_C_prior, min_skill_enforcement,
                                                     ESL_weight, gender_weight, whitelist_weight, blacklist_weight,
                                                     motivation_weight,teamwork_weight,basis_motivation,basis_IE,basis_TWS)


                # Swap participants
                groups[g1][i1], groups[g2][i2] = groups[g2][i2], groups[g1][i1]

                if selected_algorithm == "combined":
                    #print("checkpoint 7")
                    original_g1 = groups[g1][:]
                    original_g2 = groups[g2][:]

                    fitness_after1 = compute_fitness([original_g1], skill_cap, min_max_skill_A_prior,min_max_skill_B_prior,min_max_skill_C_prior, min_skill_enforcement,
                                                    ESL_weight, gender_weight, whitelist_weight, blacklist_weight,
                                                    motivation_weight,teamwork_weight,basis_motivation,basis_IE,basis_TWS)
                    fitness_after2 = compute_fitness([original_g2], skill_cap,min_max_skill_A_prior,min_max_skill_B_prior, min_max_skill_C_prior, min_skill_enforcement,
                                                    ESL_weight, gender_weight, whitelist_weight, blacklist_weight,
                                                    motivation_weight,teamwork_weight,basis_motivation,basis_IE,basis_TWS)
                    swap_accepted = fitness_after1 > fitness_before1 and fitness_after2 > fitness_before2

                # Evaluate new grouping
                if selected_algorithm == "local search":
                    after_whitelist_total1, after_blacklist_total1 = count_all_matches(groups[g1])
                    after_whitelist_total2, after_blacklist_total2 = count_all_matches(groups[g2])
                    motivation_std_after = np.std([p['motivation'] for p in groups[g1]]) + \
                                           np.std([p['motivation'] for p in groups[g2]])
                    sum_skill_after2 = max(p['skill_A'] for p in groups[g2]) + \
                                       max(p['skill_B'] for p in groups[g2]) + \
                                       max(p['skill_C'] for p in groups[g2])
                    sum_skill_after1 = max(p['skill_A'] for p in groups[g1]) + \
                                       max(p['skill_B'] for p in groups[g1]) + \
                                       max(p['skill_C'] for p in groups[g1])
                    diversity_penalty_after = diversity_penalty(groups[g1], ESL_weight, gender_weight) + diversity_penalty(
                        groups[g2], ESL_weight, gender_weight)

                    after_teamwork_std = np.std([
                        np.mean([p['teamwork'] for p in group])
                        for group in groups
                        if len(group) >= 2
                    ]) if any(len(group) >= 2 for group in groups) else 0
                #print("Check point 5")
                if selected_algorithm == "local search":
                    if group_size >= 4:
                        swap_accepted = (
                                sum_skill_after2 >= skill_cap and
                                sum_skill_after1 >= skill_cap and
                                after_whitelist_total1 + after_whitelist_total2 >= before_whitelist_total1 + before_whitelist_total2 and
                                after_blacklist_total1 + after_blacklist_total2 <= before_blacklist_total1 + before_blacklist_total2 and
                                motivation_std_before > motivation_std_after and
                                after_teamwork_std < before_teamwork_std and
                                min_skill_enforcement * min_max_skill_A_prior <= max(p['skill_A'] for p in groups[g2]) and
                                min_skill_enforcement * min_max_skill_B_prior <= max(p['skill_B'] for p in groups[g2]) and
                                min_skill_enforcement * min_max_skill_C_prior <= max(p['skill_C'] for p in groups[g2]) and
                                min_skill_enforcement * min_max_skill_A_prior <= max(p['skill_A'] for p in groups[g1]) and
                                min_skill_enforcement * min_max_skill_B_prior <= max(p['skill_B'] for p in groups[g1]) and
                                min_skill_enforcement * min_max_skill_C_prior <= max(p['skill_C'] for p in groups[g1]) and
                                diversity_penalty_after <= diversity_penalty_before)
                    else:
                        swap_accepted = (
                                sum_skill_after2 >= skill_cap and
                                sum_skill_after1 >= skill_cap and
                                after_whitelist_total1+after_whitelist_total2 >= before_whitelist_total1+before_whitelist_total2 and
                                after_blacklist_total1 + after_blacklist_total2 <= before_blacklist_total1 + before_blacklist_total2 and
                                motivation_std_before > motivation_std_after and
                                min_skill_enforcement * min_max_skill_A_prior <= max(p['skill_A'] for p in groups[g2]) and
                                min_skill_enforcement * min_max_skill_B_prior <= max(p['skill_B'] for p in groups[g2]) and
                                min_skill_enforcement * min_max_skill_C_prior <= max(p['skill_C'] for p in groups[g2]) and
                                min_skill_enforcement * min_max_skill_A_prior <= max(p['skill_A'] for p in groups[g1]) and
                                min_skill_enforcement * min_max_skill_B_prior <= max(p['skill_B'] for p in groups[g1]) and
                                min_skill_enforcement * min_max_skill_C_prior <= max(p['skill_C'] for p in groups[g1]))

                if not swap_accepted:
                    # Revert swap
                    groups[g1][i1], groups[g2][i2] = groups[g2][i2], groups[g1][i1]

                # Early stopping logic
                #print("Check point 6")
                if iteration in checkpoints:
                    avg_motivation_std = np.mean([np.std([p['motivation'] for p in group]) for group in groups])
                    motivation_std_records.append((iteration, avg_motivation_std))
                    print(f"Iteration {iteration}: Avg Motivation Std = {avg_motivation_std:.4f}")
                    print(f"Iteration {iteration}: fitness score = {compute_fitness(groups, skill_cap, min_max_skill_A_prior,min_max_skill_B_prior, min_max_skill_C_prior, min_skill_enforcement, ESL_weight, gender_weight, whitelist_weight, blacklist_weight,motivation_weight,teamwork_weight,basis_motivation,basis_IE,basis_TWS):.4f}")
                    if prev_avg_motivation_std is not None and np.isclose(avg_motivation_std, prev_avg_motivation_std,
                                                                          atol=1e-6):
                        no_change_count += 1
                    else:
                        no_change_count = 0
                    prev_avg_motivation_std = avg_motivation_std
                    if no_change_count >= 3:
                        print(f"Early stopping at iteration {iteration} after {no_change_count} unchanged checkpoints.")
                        break

        # Genetic algo
        elif selected_algorithm == "genetic":
            best_groups = genetic_algorithm(
                participants,
                num_groups,
                group_size,
                skill_cap,
                min_max_skill_A_prior,
                min_max_skill_B_prior,
                min_max_skill_C_prior,
                min_skill_enforcement,
                population_size,
                max_generations, groups_sorted_C,
                ESL_weight,
                gender_weight,
                whitelist_weight,
                blacklist_weight,
                motivation_weight,
                teamwork_weight,basis_motivation,basis_IE,basis_TWS)

            groups = best_groups
        else:
            print("Algo error")

        #done_callback()
        print(groups)
        print("Minimum total relavent skill found:", compute_max_skill(groups))
        print("Motivation score for each participant within each group")
        for idx, group in enumerate(groups):
            motivation = [p['motivation'] for p in group]
            avg = np.mean(motivation)
            std = np.std(motivation)
            print(f"Group {idx + 1}: avg={avg:.2f}, std={std:.2f}, motivation={motivation}")
        print("Introversion/Extroversion score for each participant within each group")
        for idx, group in enumerate(groups):
            ie = [p['IE'] for p in group]
            avg = np.mean(ie)
            std = np.std(ie)
            print(f"Group {idx + 1}: avg={avg:.2f}, std={std:.2f}, I/E={ie}")

        # MAX SKILL IN EACH GROUP AFTER MOTIVATION OPTIMIZATION
        print("")
        print("Max relavent skill within each group")
        for i, group in enumerate(groups, start=1):
            max_skill_A = max(p['skill_A'] for p in group)
            max_skill_B = max(p['skill_B'] for p in group)
            max_skill_C = max(p['skill_C'] for p in group)

            print(f"Group {i}: max skill_A = {max_skill_A}, max skill_B = {max_skill_B}, max skill_C = {max_skill_C}")
        print("")
        print("Minimum Maximum relavent skill score within each group")
        for group in groups:
            max_skill_A = max(p['skill_A'] for p in group)
            max_skill_B = max(p['skill_B'] for p in group)
            max_skill_C = max(p['skill_C'] for p in group)

            max_skills_A.append(max_skill_A)
            max_skills_B.append(max_skill_B)
            max_skills_C.append(max_skill_C)

        # Find the minimum among the group max values for each skill
        min_max_skill_A = min(max_skills_A)
        min_max_skill_B = min(max_skills_B)
        min_max_skill_C = min(max_skills_C)

        print('Minimum skill A level in each group', min_max_skill_A)
        print('Minimum skill B level in each group', min_max_skill_B)
        print('Minimum skill C level in each group', min_max_skill_C)
        # percentage of groups with lone ESL/gender
        percent_lone_gender, percent_lone_esl, lone_gender_group_ids, lone_esl_group_ids = count_lone_groups(groups)
        print(f"Percentage of groups with lone gender participant: {percent_lone_gender:.2f}%")
        print(f"Percentage of groups with lone ESL participant: {percent_lone_esl:.2f}%")
        print("Groups with lone gender participant:", lone_gender_group_ids)
        print("Groups with lone ESL participant:", lone_esl_group_ids)
        print(sum(len(group) for group in groups))
        for i, group in enumerate(groups):
            total_whitelist, total_blacklist = count_all_matches(group)
            print(f"Group {i}: Total whitelist matches: {total_whitelist}, Total blacklist matches: {total_blacklist}")

        end_time = time.time()  # End timer

        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")

        # format data to excel
        output = io.BytesIO()
        rows = []
        for i, group in enumerate(groups, start=1):
            for member in group:
                row = {
                    "Group Number": i
                }
                if show_names:
                    row["Name"] = member['name']
                if show_email:
                    row["Email"] = member['email']
                if show_gender:
                    row["Gender"] = member['gender']
                if show_ESL:
                    row["ESL"] = member['ESL']
                rows.append(row)

        df = pd.DataFrame(rows)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            ws = writer.sheets['Sheet1']
            # Adjust columns widths if needed, e.g.:
            ws.column_dimensions['A'].width = 15
            ws.column_dimensions['B'].width = 30
            ws.column_dimensions['C'].width = 40

        output.seek(0)
        with open('static/output/groups.xlsx', 'wb') as f:
            f.write(output.getvalue())

        # Direct Excel file download

        return output, groups

    except Exception as e:
        return jsonify({'error': str(e)}), 400
        print("checkpoint 7")

    except Exception as e:
        print("Error1", e)
        traceback.print_exc()




#post processing
def add_page_number(canvas, doc):
    page_num = canvas.getPageNumber()
    text = f"Page {page_num}"
    # Position it at the bottom center
    canvas.setFont("Helvetica", 9)
    canvas.drawCentredString(300, 15, text)


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pages = []

    def showPage(self):
        self.pages.append(dict(self.__dict__))  # Save the current page state
        self._startPage()

    def save(self):
        page_count = len(self.pages)
        for page in self.pages:
            self.__dict__.update(page)
            self.draw_page_number(page_count)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.setFont("Helvetica", 9)
        self.drawRightString(7.5 * inch, 0.75 * inch,
                             f"Page {self._pageNumber} of {page_count}")

#chaange the TOC link to top of page
class MyHeading(Paragraph):
    def __init__(self, text, style, bookmark_name, toc_level):
        super().__init__(text, style)
        self._bookmarkName = bookmark_name
        self.toc_level = toc_level

    def draw(self):
        # Place bookmark *before* drawing heading text
        self.canv.bookmarkPage(self._bookmarkName)
        super().draw()



def generate_reportlab_pdf(groups, num_participants, group_size,filename, elapsed_time, skill_A_name, skill_B_name, skill_C_name, selected_algorithm):
    #variables


    motivation_avgs = []
    motivation_stds = []
    ie_avgs = []
    ie_stds = []
    TW_avgs=[]
    TW_stds=[]

    def heading_with_toc(text, style, level):
        para = Paragraph(text, style)
        para.toc_level = level
        # clean bookmark name: replace spaces and punctuation with underscores, lowercase
        clean_name = ''.join(c if c.isalnum() else '_' for c in text).lower()
        para._bookmarkName = clean_name
        return para

    try:
        print(f"Groups: {groups}")
        print(f"Num participants: {num_participants}, Group size: {group_size}, time: {elapsed_time:.4f} seconds")

    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        raise

    if filename is None or isinstance(filename, io.BytesIO):
        output = filename or io.BytesIO()
    else:
        output = open(filename, 'wb')


    # Setup PDF document
    output = io.BytesIO()

    doc = BaseDocTemplate(output, pagesize=letter,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)


    styles = getSampleStyleSheet()
    styles['Heading1'].fontSize = 14
    styles['Heading1'].leading = 16
    styles['Heading1'].spaceAfter = 12

    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id='normal')

    doc.addPageTemplates([PageTemplate(id='main', frames=frame, onPage=add_page_number)])

    def after_flowable(flowable):
        if hasattr(flowable, 'toc_level') and hasattr(flowable, '_bookmarkName'):
            level = flowable.toc_level
            try:
                text = flowable.getPlainText()
            except Exception:
                text = str(flowable)
            key = flowable._bookmarkName
            # Notify TOC with key for clickable links
            doc.notify('TOCEntry', (level, text, doc.page, key))
            # Add bookmark (outline) entry in PDF
            doc.canv.bookmarkPage(key)
            doc.canv.addOutlineEntry(text, key, level=level, closed=False)
        else:
            pass  # flowables without toc_level or bookmark don't affect TOC
    doc.afterFlowable = after_flowable
    story = []


    # Title
    story.append(Paragraph("Group Allocation Report", styles['Title']))
    story.append(Spacer(1, 12))

    #table of content
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(fontSize=14, name='TOCLevel1', leftIndent=20, firstLineIndent=-20, spaceBefore=5),
        ParagraphStyle(fontSize=12, name='TOCLevel2', leftIndent=20, firstLineIndent=-20, spaceBefore=5),
    ]
    story.append(Paragraph("Table of Contents", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(toc)
    story.append(PageBreak())

    def add_section(heading1, subheadings):
        # Heading1: level 0
        story.append(heading_with_toc(heading1, styles['Heading1'], level=0))
        story.append(Spacer(1, 12))

        for sub in subheadings:
            # Heading2: level 1
            para = heading_with_toc(sub, styles['Heading2'], level=1)
            story.append(para)
            story.append(Spacer(1, 6))
            for i in range(5):
                story.append(Paragraph(f"Text under {sub}, paragraph {i + 1}", styles['Normal']))
                story.append(Spacer(1, 4))
            story.append(Spacer(1, 12))

    # Add chart image


    chart_path = create_motivation_ie_chart(groups)
    story.append(heading_with_toc("Motivation & I/E Averages Visualization", styles['Heading2'], level=0))

    story.append(Image(chart_path, width=500, height=250))
    story.append(Spacer(1, 12))


    # Minimum total relevant skill
    story.append(Spacer(1, 12))

    # Motivation Scores section
    #add_section("Motivation scores for each group",["Subsection 1.1", "Subsection 1.2"])
    story.append(heading_with_toc("Motivation scores for each group", styles['Heading2'], level=0))

    for idx, group in enumerate(groups):
        motivation = [p['motivation'] for p in group]
        avg = np.mean(motivation)
        std = np.std(motivation)
        motivation_avgs.append(avg)
        motivation_stds.append(std)
        text = f"Group {idx + 1}: mean = {avg:.2f}, std = {std:.2f}, motivations = {motivation}"
        story.append(Paragraph(text, styles['Normal']))
    story.append(Spacer(1, 12))

    overall_motivation_avg = np.mean(motivation_avgs)
    overall_motivation_std_avg = np.mean(motivation_stds)
    story.append(Paragraph(f"<b>Mean motivation score between all groups:</b> {overall_motivation_avg:.2f}", styles['Normal']))
    story.append(Paragraph(f"<b>Mean standard deviation across all groups:</b> {overall_motivation_std_avg:.2f}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Introversion/Extroversion Scores

    story.append(heading_with_toc("Introversion/Extroversion Scores", styles['Heading2'], level=0))

    for idx, group in enumerate(groups):
        ie = [p['IE'] for p in group]
        avg = np.mean(ie)
        std = np.std(ie)
        ie_avgs.append(avg)
        ie_stds.append(std)
        text = f"Group {idx + 1}: mean = {avg:.2f}, std = {std:.2f}, I/E = {ie}"
        story.append(Paragraph(text, styles['Normal']))
    story.append(Spacer(1, 12))

    overall_ie_avg = np.mean(ie_avgs)
    overall_ie_std_avg = np.mean(ie_stds)
    story.append(Paragraph(f"<b>Mean introversion/extroversion score between all groups:</b> {overall_ie_avg:.2f}", styles['Normal']))
    story.append(Paragraph(f"<b>Mean standard deviation between all groups:</b> {overall_ie_std_avg:.2f}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Max skill scores in each group
    plot_folder = 'temp_plots'
    os.makedirs(plot_folder, exist_ok=True)

    plot_skills_bar_chart(groups=groups,filename_max=os.path.join(plot_folder, 'max_skills.png'),
                          filename_avg=os.path.join(plot_folder, 'avg_skills.png'), skill_A_name=skill_A_name, skill_B_name=skill_B_name, skill_C_name=skill_C_name)

    # Add graphs to PDF :

    story.append(heading_with_toc("Maximum Skill Scores per Group", styles['Heading2'], level=0))

    story.append(Image(os.path.join(plot_folder, 'max_skills.png'), width=500, height=250))
    story.append(Spacer(1, 12))


    story.append(heading_with_toc("Average skill scores per group", styles['Heading2'], level=0))

    story.append(Image(os.path.join(plot_folder, 'avg_skills.png'), width=500, height=250))
    story.append(Spacer(1, 12))

    story.append(heading_with_toc("Maximum Skill Scores in Each Group", styles['Heading2'], level=0))

    max_skills_A, max_skills_B, max_skills_C = [], [], []
    for i, group in enumerate(groups, start=1):
        max_A = max(p['skill_A'] for p in group)
        max_B = max(p['skill_B'] for p in group)
        max_C = max(p['skill_C'] for p in group)
        max_skills_A.append(max_A)
        max_skills_B.append(max_B)
        max_skills_C.append(max_C)
        text = f"Group {i}: max {skill_A_name} = {max_A}, max {skill_B_name} = {max_B}, max {skill_C_name} = {max_C}"
        story.append(Paragraph(text, styles['Normal']))
    story.append(Spacer(1, 12))

    # Minimum of maximum skills across groups
    story.append(heading_with_toc("Minimum of Maximum Skills Across Groups", styles['Heading2'], level=0))

    story.append(Paragraph(f"Min {skill_A_name}: {min(max_skills_A)}", styles['Normal']))
    story.append(Paragraph(f"Min {skill_B_name}: {min(max_skills_B)}", styles['Normal']))
    story.append(Paragraph(f"Min {skill_C_name}: {min(max_skills_C)}", styles['Normal']))
    story.append(Spacer(1, 12))


    #global average of skill
    all_participants = [p for group in groups for p in group]

    # Extract all skill values
    all_skill_A = [p['skill_A'] for p in all_participants]
    all_skill_B = [p['skill_B'] for p in all_participants]
    all_skill_C = [p['skill_C'] for p in all_participants]

    # Compute mean and median
    mean_A = statistics.mean(all_skill_A)
    median_A = statistics.median(all_skill_A)

    mean_B = statistics.mean(all_skill_B)
    median_B = statistics.median(all_skill_B)

    mean_C = statistics.mean(all_skill_C)
    median_C = statistics.median(all_skill_C)

    # Add section to PDF
    story.append(heading_with_toc("Global averages of skills", styles['Heading2'], level=0))


    story.append(Paragraph(f"{skill_A_name} - Mean: {mean_A:.2f}, Median: {median_A:.2f}", styles['Normal']))
    story.append(Paragraph(f"{skill_B_name} - Mean: {mean_B:.2f}, Median: {median_B:.2f}", styles['Normal']))
    story.append(Paragraph(f"{skill_C_name} - Mean: {mean_C:.2f}, Median: {median_C:.2f}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Lone participants section
    lone_gender, lone_esl, lone_gender_groups, lone_esl_groups = count_lone_groups(groups)
    story.append(heading_with_toc("Lone participants", styles['Heading2'], level=0))


    story.append(Paragraph(f"Percentage of groups with lone gender participant: {lone_gender:.2f}%", styles['Normal']))
    story.append(Paragraph(f"Groups with lone gender participant: {', '.join(map(str, lone_gender_groups)) or 'None'}",
                           styles['Normal']))
    story.append(Paragraph(f"Percentage of groups with lone ESL participant: {lone_esl:.2f}%", styles['Normal']))
    story.append(Paragraph(f"Groups with lone ESL participant: {', '.join(map(str, lone_esl_groups)) or 'None'}",
                           styles['Normal']))
    story.append(Spacer(1, 12))

    # Count total participants
    total = len(all_participants)

    # Count ESL
    esl_yes = sum(1 for p in all_participants if p.get('ESL', '').lower() == 'yes')
    esl_no = total - esl_yes
    # Count gender
    gender_counts = {'male': 0, 'female': 0, 'other': 0}
    for p in all_participants:
        gender = p.get('gender', '').lower()
        if gender in gender_counts:
            gender_counts[gender] += 1
        else:
            gender_counts['other'] += 1  # Treat unknowns as 'other'

    # ESL percentages
    esl_yes_pct = (esl_yes / total) * 100 if total else 0
    esl_no_pct = (esl_no / total) * 100 if total else 0

    # Gender percentages
    male_pct = (gender_counts['male'] / total) * 100 if total else 0
    female_pct = (gender_counts['female'] / total) * 100 if total else 0
    other_pct = (gender_counts['other'] / total) * 100 if total else 0
    #list ESL and gender
    for idx, group in enumerate(groups):
        # Get ESL and gender values for each participant
        esl_values = [p.get('ESL', 'N/A') for p in group]
        gender_values = [p.get('gender', 'N/A') for p in group]

        # ESL list
        esl_text = f"Group {idx + 1}: ESL = {esl_values}"
        story.append(Paragraph(esl_text, styles['Normal']))

        # Gender list
        gender_text = f"Group {idx + 1}: Gender = {gender_values}"
        story.append(Paragraph(gender_text, styles['Normal']))

    story.append(Spacer(1, 12))

    # Add section to report
    story.append(heading_with_toc("Global demographics", styles['Heading2'], level=0))


    story.append(Paragraph(f"ESL - Yes: {esl_yes_pct:.2f}%, No: {esl_no_pct:.2f}%", styles['Normal']))
    story.append(Paragraph(f"Gender - Male: {male_pct:.2f}%, Female: {female_pct:.2f}%, Other: {other_pct:.2f}%",
                           styles['Normal']))
    story.append(Spacer(1, 12))
    #teamwork score
    story.append(heading_with_toc("Teamwork score", styles['Heading2'], level=0))

    chart_path = TW_chart(groups)
    story.append(Image(chart_path, width=500, height=250))
    story.append(Spacer(1, 12))

    for idx, group in enumerate(groups):
        TWS = [p['teamwork'] for p in group]
        avg = np.mean(TWS)
        std = np.std(TWS)
        TW_avgs.append(avg)
        TW_stds.append(std)
        text = f"Group {idx + 1}: mean = {avg:.2f}, std = {std:.2f}, Teamwork = {TWS}"
        story.append(Paragraph(text, styles['Normal']))
    story.append(Spacer(1, 12))

    overall_motivation_avg = np.mean(TW_avgs)
    overall_motivation_std_avg = np.mean(TW_stds)
    story.append(Paragraph(f"<b>Mean teamwork score between all groups:</b> {overall_motivation_avg:.2f}", styles['Normal']))
    story.append(Paragraph(f"<b>Mean standard deviation across all groups:</b> {overall_motivation_std_avg:.2f}", styles['Normal']))
    story.append(Spacer(1, 12))




    #Whitelist and blacklist


    story.append(heading_with_toc("Whitelist/Blacklist Matches per Group", styles['Heading2'], level=0))

    total_whitelist_all = 0
    total_blacklist_all = 0
    num_groups = len(groups)

    for idx, group in enumerate(groups):
        total_whitelist, total_blacklist = count_all_matches(group)
        total_whitelist_all += total_whitelist
        total_blacklist_all += total_blacklist
        text = (f"Group {idx + 1}: Total whitelist matches = {total_whitelist}, "
                f"Total blacklist matches = {total_blacklist}")
        story.append(Paragraph(text, styles['Normal']))

    story.append(Spacer(1, 12))

    avg_whitelist = total_whitelist_all / num_groups if num_groups else 0
    avg_blacklist = total_blacklist_all / num_groups if num_groups else 0

    summary_text = (f"Total whitelist matches across all groups = {total_whitelist_all}, "
                    f"Total blacklist matches across all groups = {total_blacklist_all}<br/>"
                    f"Average whitelist matches per group = {avg_whitelist:.2f}, "
                    f"Average blacklist matches per group = {avg_blacklist:.2f}")

    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 12))



    #solver statistics
    story.append(heading_with_toc("Solver statistics", styles['Heading2'], level=0))


    # Total participants
    story.append(Paragraph(f"<b> Algorithm:</b> {selected_algorithm}", styles['Normal']))
    total_participants = sum(len(g) for g in groups)
    story.append(Paragraph(f"<b>Total number of participants:</b> {total_participants}", styles['Normal']))
    story.append(Paragraph(
        f"<b>Total possible combinations for grouping:</b> {ways_with_leftovers(num_participants, group_size):.3e}",
        styles['Normal']))
    story.append(Paragraph(
        f"<b>Total runtime:</b> {elapsed_time} seconds",
        styles['Normal']))



    # Build PDF
    doc.multiBuild(story)
    if os.path.exists(chart_path):
        os.remove(chart_path)

    # save buffer to disk if filename is provided
    if filename and not isinstance(filename, io.BytesIO):
        with open(filename, 'wb') as f:
            f.write(output.getvalue())
        output.close()

    output.seek(0)
    return output




def plot_skills_bar_chart(groups, filename_max, filename_avg, skill_A_name, skill_B_name, skill_C_name):
    groups_list = list(range(1, len(groups) + 1))

    max_skills_A, max_skills_B, max_skills_C = [], [], []
    avg_skills_A, avg_skills_B, avg_skills_C = [], [], []

    for group in groups:
        skill_A_vals = [p['skill_A'] for p in group]
        skill_B_vals = [p['skill_B'] for p in group]
        skill_C_vals = [p['skill_C'] for p in group]

        max_skills_A.append(max(skill_A_vals))
        max_skills_B.append(max(skill_B_vals))
        max_skills_C.append(max(skill_C_vals))

        avg_skills_A.append(np.mean(skill_A_vals))
        avg_skills_B.append(np.mean(skill_B_vals))
        avg_skills_C.append(np.mean(skill_C_vals))

    width = 0.2
    x = np.arange(len(groups))

    # plot max skills chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, max_skills_A, width, label=f'Max {skill_A_name}')
    ax.bar(x, max_skills_B, width, label=f'Max {skill_B_name}')
    ax.bar(x + width, max_skills_C, width, label=f'Max {skill_C_name}')

    ax.set_xlabel('Group')
    ax.set_ylabel('Max Skill Score')
    ax.set_title('Maximum Skill Scores per Group')
    ax.set_xticks(x)
    ax.set_xticklabels(groups_list)
    ax.legend()
    plt.tight_layout()
    fig.savefig(filename_max)
    plt.close(fig)

    # plot average skills chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, avg_skills_A, width, label=f'Mean {skill_A_name} score')
    ax.bar(x, avg_skills_B, width, label=f'Mean {skill_B_name} score')
    ax.bar(x + width, avg_skills_C, width, label=f'Mean {skill_C_name} score')

    ax.set_xlabel('Group')
    ax.set_ylabel('Average Skill Score')
    ax.set_title('Average Skill Scores per Group')
    ax.set_xticks(x)
    ax.set_xticklabels(groups_list)
    ax.legend()
    plt.tight_layout()
    fig.savefig(filename_avg)
    plt.close(fig)






def create_motivation_ie_chart(groups):
    group_labels = [f'{i+1}' for i in range(len(groups))]
    motivation_avgs = [np.mean([p['motivation'] for p in g]) for g in groups]
    ie_avgs = [np.mean([p['IE'] for p in g]) for g in groups]

    x = range(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width/2 for i in x], motivation_avgs, width, label='Motivation Avg')
    ax.bar([i + width/2 for i in x], ie_avgs, width, label='I/E Avg')

    ax.set_xlabel('Groups')
    ax.set_ylabel('Average Score')
    ax.set_title('Motivation and I/E Averages by Group')
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    # Save to a temporary file and return the path
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.tight_layout()
    plt.savefig(temp_file.name)
    plt.close()
    return temp_file.name


def TW_chart(groups):
    group_labels = [f'{i+1}' for i in range(len(groups))]
    TW_avgs = [np.mean([p['teamwork'] for p in g]) for g in groups]


    x = range(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width/2 for i in x], TW_avgs, width, label='Teamwork score Avg')


    ax.set_xlabel('Groups')
    ax.set_ylabel('Average Teamwork Score')
    ax.set_title('Teamwork Averages by Group')
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    # Save to a temporary file and return the path
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.tight_layout()
    plt.savefig(temp_file.name)
    plt.close()
    return temp_file.name



@app.route('/run_allocation', methods=['POST'])
def run_allocation():
    print('Form data:', request.form)
    print('Files:', request.files)
    try:
        def handle_uploaded_file():
            # Get uploaded file
            file = request.files['file']
            filename = secure_filename(file.filename)

            # Save the file temporarily
            upload_folder = app.config.get('UPLOAD_FOLDER', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)

            # load file into pandas DataFrame
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    import openpyxl  # import for CSV
                    df = pd.read_excel(file_path, engine='openpyxl')

                if df.empty:
                    raise ValueError("The selected file is empty or unreadable.")

                return df  # returns loaded DataFrame

            except Exception as e:
                raise RuntimeError(f"Failed to load file: {str(e)}")

        # Parse the JSON string from 'settings'
        settings_json = request.form.get('settings')
        if not settings_json:
            return jsonify(error="Missing settings data"), 400

        try:
            settings = json.loads(settings_json)
        except json.JSONDecodeError:
            return jsonify(error="Invalid JSON in settings"), 400

        try:

            df = handle_uploaded_file()


        except Exception as e:
            return jsonify(error=str(e)), 400
        # acessing settings
        num_participants = settings.get('num_participants', 0)
        group_size = settings.get('group_size', 0)
        algorithm = settings.get('algorithm', 'local search')
        max_iterations = settings.get('max_iterations', 50000)
        max_generations = settings.get('max_generations', 50)
        population_size = settings.get('population_size', 100)
        motivation_weight = float(settings.get('motivation_weight', 1))
        academic_weight = float(settings.get('academic_weight', 0.5))
        whitelist_weight = float(settings.get('whitelist_weight', 1))
        blacklist_weight = float(settings.get('blacklist_weight', 2))
        teamwork_weight = float(settings.get('teamwork_weight', 0.2))
        whitelist_limit = int(settings.get('whitelist_limit', 2))
        blacklist_limit = int(settings.get('blacklist_limit', 2))
        normalise_mode = bool(settings.get('normalise_skill', False))
        print(f"normalise mode: {normalise_mode}")
        ESL_weight = float(settings.get('ESL_weight', 0.8))
        skill_weight = float(settings.get('skill_weight', 0.9))
        min_skill = float(settings.get('min_skill', 0.8))
        gender_weight = float(settings.get('gender_weight', 0.8))
        rounding_mode = bool(settings.get('rounding_mode', True))
        selected_algorithm = settings.get('algorithm', 'local search')

        show_names = bool(settings.get('show_names', True))
        show_email = bool(settings.get('show_email', True))
        show_gender = bool(settings.get('show_gender', False))
        show_ESL = bool(settings.get('show_ESL', False))
        show_report = bool(settings.get('show_report', True))

        # Convert numbers to int or float
        try:
            num_participants = int(num_participants)
            group_size = int(group_size)
            max_iterations = int(max_iterations)
            max_generations = int(max_generations)
            population_size = int(population_size)
        except (ValueError, TypeError):
            traceback.print_exc()
            return jsonify(error="Invalid number format in settings"), 400

        mapping = {
            'strongly disagree': 1,
            'disagree': 2,
            'neutral': 3,
            'agree': 4,
            'strongly agree': 5
        }



        print(f"1inital num participants: {num_participants}")
        print(f"1inital group size: {group_size}")
        def map_responses(val):
            if isinstance(val, (int, float)) and 1 <= val <= 5:
                return val
            return mapping.get(str(val).strip().lower(), val)


        #CSV robustness check
        df['skill 1'] = [0] * len(df)
        df['skill 2'] = [0] * len(df)
        df['skill 3'] = [0] * len(df)
        df['grade']=[0] * len(df)
        df['introvert extrovert']=[0] * len(df)
        df['gender'] = ["Other"] * len(df)
        df['english'] = ["No"] * len(df)



        pd.set_option('display.max_columns', None)  # Show all columns
        print(df.head(3))
        #data fram heading extraction


        columns = df.columns
        email_col = next(col for col in columns if 'email' in col.lower())
        gender_col = next(col for col in columns if 'gender' in col.lower())
        english_col = next(col for col in columns if 'english' in col.lower())
        grade_col = next(col for col in columns if 'grade' in col.lower())

        score_col = pd.to_numeric(df[grade_col].astype(str).str[0], errors='coerce')

        IE_col = next(
            col for col in columns
            if any(term in col.lower() for term in ['introvert', 'extrovert'])
        )
        df[IE_col] = df[IE_col].apply(map_responses)
        #teamwork
        teamwork_cols = [col for col in columns if 'teamwork' in col.lower()]
        print(teamwork_cols)
        #1 3 4 5 6 7 8 10 11 14

        mapping_teamwork1 = {
            'Very frequently': 1,
            'Frequently': 2,
            'Sometimes': 3,
            'Rarely': 2,
            'Never': 1
        }

        mapping_teamwork3 = {
            'Very frequently': 1,
            'Frequently': 2,
            'Sometimes': 3,
            'Rarely': 2,
            'Never': 1
        }

        mapping_teamwork4 = {
            'Very frequently': 2,
            'Frequently': 2,
            'Sometimes': 3,
            'Rarely': 1,
            'Never': 0
        }

        mapping_teamwork5 = {
            'Very frequently': 0,
            'Frequently': 1,
            'Sometimes': 3,
            'Rarely': 1,
            'Never': 0
        }

        mapping_teamwork6 = {
            'Very frequently': 3,
            'Frequently': 3,
            'Sometimes': 2,
            'Rarely': 1,
            'Never': 0
        }

        mapping_teamwork7 = {
            'Very frequently': 3,
            'Frequently': 3,
            'Sometimes': 2,
            'Rarely': 1,
            'Never': 0
        }

        mapping_teamwork8 = {
            'Very frequently': 2,
            'Frequently': 3,
            'Sometimes': 3,
            'Rarely': 1,
            'Never': 0
        }
        mapping_teamwork10 = {
            'Very frequently': 0,
            'Frequently': 0,
            'Sometimes': 3,
            'Rarely': 2,
            'Never': 1
        }

        mapping_teamwork11 = {
            'Very frequently': 2,
            'Frequently': 3,
            'Sometimes': 3,
            'Rarely': 1,
            'Never': 0
        }
        mapping_teamwork14 = {
            'Very frequently': 2,
            'Frequently': 2,
            'Sometimes': 3,
            'Rarely': 1,
            'Never': 0
        }
        mappings = {
            'teamwork1': mapping_teamwork1,
            'teamwork3': mapping_teamwork3,
            'teamwork4': mapping_teamwork4,
            'teamwork5': mapping_teamwork5,
            'teamwork6': mapping_teamwork6,
            'teamwork7': mapping_teamwork7,
            'teamwork8': mapping_teamwork8,
            'teamwork10': mapping_teamwork10,
            'teamwork11': mapping_teamwork11,
            'teamwork14': mapping_teamwork14,

        }

        def get_base_key(col_name):
            return col_name.split(':')[0].strip()

        for col in teamwork_cols:
            base_key = get_base_key(col).lower()
            if base_key in mappings:
                df[col + '_score'] = df[col].apply(
                    lambda x: mappings[base_key].get(str(x).strip(), 0)
                )

            else:
                df[col + '_score'] = 0

        TWscore_cols = [col + '_score' for col in teamwork_cols if get_base_key(col).lower() in mappings]



        # Sum across those columns row-wise, result in new column 'teamwork'
        df['teamwork'] = df[TWscore_cols].sum(axis=1)

        #print("Teamwork score columns:", TWscore_cols)
        #print(df[['teamwork'] + TWscore_cols].head())

        skill_cols = [col for col in columns if 'skill' in col.lower()][:3]

        # Store first three skill column names individually
        skill_A_name = skill_cols[0] if len(skill_cols) > 0 else None
        skill_B_name = skill_cols[1] if len(skill_cols) > 1 else None
        skill_C_name = skill_cols[2] if len(skill_cols) > 2 else None

        skill_A, skill_B, skill_C = skill_cols[:3]
        print(f"skill A col= {skill_A}")

        # Find all columns that contain the word 'whitelist'
        whitelist_cols = [col for col in columns if 'whitelist' in col.lower()]

        print(f"whitelist cols = {whitelist_cols}")

        # For each row, collect whitelist values from all whitelist columns
        whitelists = []
        for _, row in df.iterrows():
            row_whitelist = []
            for col in whitelist_cols:
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    row_whitelist.append(str(val).strip())
            whitelists.append(row_whitelist)

        blacklist_cols = [col for col in columns if 'blacklist' in col.lower()]

        blacklists = []
        for _, row in df.iterrows():
            row_blacklist = []
            for col in blacklist_cols:
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    row_blacklist.append(str(val).strip())
            blacklists.append(row_blacklist)

        personality_cols = [
            col for col in columns
            if 'personality' in col.lower() and 'introvert' not in col.lower() and 'extrovert' not in col.lower()
        ]
        for col in personality_cols:
            df[col] = df[col].apply(map_responses)


        df['Personality Average'] = df[personality_cols].astype(float).mean(axis=1)
        df['motivation'] = ((1 - academic_weight) * df['Personality Average'] +
                            academic_weight * score_col)

        df['motivation']=(5/df['motivation'].max())*df['motivation']


        print("Before normalization:")
        print(df[[skill_A, skill_B, skill_C]].head())
        print(f"normalise mode: {normalise_mode}")
        if normalise_mode:
            def normalize_and_score_row(row):
                # Get original skill values
                original = row[[skill_A, skill_B, skill_C]].values.astype(float)

                min_val = original.min()
                max_val = original.max()

                if max_val > min_val:
                    # Normalization (0 to 10)
                    normalized = (original - min_val) / (max_val - min_val)
                    # Final score = original + normalized * 5
                    final = original + (normalized * 5)
                else:
                    # All skills are equal, use 0.33 * 5 instead of normalization
                    final = original + (0.33 * 5)

                return pd.Series(final, index=[skill_A, skill_B, skill_C])

            df[[skill_A, skill_B, skill_C]] = df.apply(normalize_and_score_row, axis=1)

        participants = []
        for idx, row in df.iterrows():
            participants.append({
                'id': idx + 1,
                'name': row['Name'],
                'email': row[email_col],
                'gender': row[gender_col],
                'ESL': row[english_col],
                'skill_A': row[skill_A],
                'skill_B': row[skill_B],
                'skill_C': row[skill_C],
                'personality': round(row['Personality Average'], 2),
                'motivation': round(row['motivation'], 2),
                'IE': row[IE_col],
                'teamwork': round(row['teamwork'],2),
                'score': score_col.iloc[idx],
                'whitelist': whitelists[idx],
                'blacklist' : blacklists[idx]
            })


        #fix nan or errors
        for p in participants:
            # If personality is NaN or not a number, set to 0
            if not isinstance(p['personality'], (int, float)) or pd.isna(p['personality']):
                p['personality'] = 0.0

            # If motivation is NaN or not a number, set to 0
            if not isinstance(p['motivation'], (int, float)) or pd.isna(p['motivation']):
                p['motivation'] = 0.0

            if not isinstance(p['ESL'], str) or not p['ESL'].strip():
                p['ESL'] = "No"
            if not isinstance(p['ESL'], str) or not p['ESL'].strip():
                p['ESL'] = "No"

        # Build email and id lookup
        email_to_id = {p['email']: p['id'] for p in participants}
        print("checkpoint 1")
        # Replace emails with IDs in whitelist and blacklist
        for p in participants:
            p['whitelist'] = [email_to_id[email] for email in p['whitelist'] if email in email_to_id][:whitelist_limit]
            p['blacklist'] = [email_to_id[email] for email in p['blacklist'] if email in email_to_id][:blacklist_limit]

        print(participants)
        print(f"inital num participants: {num_participants}")
        print(f"inital group size: {group_size}")

        try:
            print("checkpoint 2")
            excel_buffer,groups = start_allocation(
                participants,
                academic_weight,
                normalise_mode,
                ESL_weight,
                skill_weight,
                min_skill,
                gender_weight,
                selected_algorithm,
                max_iterations,
                max_generations,
                population_size,
                group_size,
                num_participants,
                rounding_mode,
                show_names,
                show_email,
                show_report,
                show_gender,
                show_ESL,
                blacklist_weight,
                whitelist_weight,
                motivation_weight,
                teamwork_weight
            )
            print("start_allocation returned:", type(excel_buffer))
            print("Type of groups:", type(groups))
            data_id = str(uuid.uuid4())
            session['data_id'] = data_id
            temp_storage[data_id] = {
                'groups': groups,
                'group_size': group_size,
                'skill_A_name': skill_A_name,
                'skill_B_name':skill_B_name,
                'skill_C_name': skill_C_name,
                'selected_algorithm':selected_algorithm
            }


        except Exception as e:
            print("Error in start_allocation:", str(e))
            return jsonify(error=f"start_allocation error: {str(e)}"), 500

        response = send_file(
            excel_buffer,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='groups.xlsx'
        )

        try:
            # --- Clean uploads ---
            uploads_folder = 'uploads'
            uploads_path = os.path.abspath(uploads_folder)
            if uploads_path.endswith('uploads'):
                for filename in os.listdir(uploads_path):
                    file_path = os.path.join(uploads_path, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

            # --- Clean static/output ---
            output_folder = 'static/output'
            output_path = os.path.abspath(output_folder)
            if output_path.endswith(os.path.join('static', 'output')):
                for filename in os.listdir(output_path):
                    file_path = os.path.join(output_path, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)



        except Exception as e:
            traceback.print_exc()
            print("Unhandled error in /run_allocation:", str(e))
            return jsonify({'error': str(e)}), 400


    except Exception as e:
        traceback.print_exc()
        print("Unhandled error in /run_allocation:", str(e))
        return jsonify({'error': str(e)}), 400
        # Return the file download response


    return response



#to download report
@app.route('/download_report')
def download_report():
    global elapsed_time
    data_id = session.get('data_id')
    data = temp_storage.get(data_id)
    if not data:
        return "No data found", 400

    groups = data['groups']
    group_size = data['group_size']
    skill_A_name= data['skill_A_name']
    skill_B_name=data['skill_B_name']
    skill_C_name=data['skill_C_name']
    selected_algorithm=data['selected_algorithm']

    num_participants=max(p["id"] for group in groups for p in group)


    try:
        pdf_buffer = generate_reportlab_pdf(groups, num_participants, group_size, filename=None,elapsed_time=elapsed_time, skill_A_name=skill_A_name, skill_B_name=skill_B_name, skill_C_name=skill_C_name, selected_algorithm=selected_algorithm)
        print("pdf_buffer type:", type(pdf_buffer))
        print("pdf_buffer:", pdf_buffer)
        if not hasattr(pdf_buffer, 'seek'):
            raise ValueError("Invalid PDF buffer returned")
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name='groups_report.pdf',
            mimetype='application/pdf'
        )
    except Exception as e:
        raise RuntimeError(f"Error generating PDF: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/mines')
def mines():
    return render_template('mines.html')
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')
@app.route('/dice')
def dice():
    return render_template('dice.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
