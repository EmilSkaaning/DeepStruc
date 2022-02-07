#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:52:34 2020

@author: andysodeanker
"""
import sys

from diffpy.Structure import Structure
from diffpy.Structure import Atom
from diffpy.Structure import loadStructure, Structure, Lattice
import numpy as np
from ase.data import covalent_radii, atomic_numbers, chemical_symbols
from ase.cluster.cubic import FaceCenteredCubic, BodyCenteredCubic, SimpleCubic
from ase.lattice.hexagonal import Hexagonal, HexagonalClosedPacked, Graphite
from ase.cluster.decahedron import Decahedron
from ase.cluster.icosahedron import Icosahedron
from ase.cluster.octahedron import Octahedron
from tqdm import tqdm
import mendeleev, datetime
import os, pdb
from shutil import copyfile

#from simPDFs import *

"""
The structure geometries are set into categories.
Sc: 0, FCC: 1, BCC: 2, HCP: 3, Icosahedron: 4, Decahedron: 5, Octahedron: 6
The  lattice constant is chossen between 90-110 % of the normal covalent_radii*2 for the atom.
"""

import shutil


def new_structure_checker(Input_array, list_search):
	for i in range(len(list_search)):
		if np.all(Input_array == list_search[i]) == True:
			return True
		else:
			pass
	return False

def find_geometry(structure_type):
	if structure_type==0: 
		Geometry = "SC"
	if structure_type==1: 
		Geometry = "FCC"
	if structure_type==2: 
		Geometry = "BCC"
	if structure_type==3:
		Geometry = "HCP"
	if structure_type==4:
		Geometry = "Ico"
	if structure_type==5:
		Geometry = "Dec"
	if structure_type==6:
		Geometry = "Oct"
	return Geometry

def find_structure_type(Geometry):
	if Geometry=="SC": 
		structure_type = 0
	if Geometry=="FCC": 
		structure_type = 1
	if Geometry=="BCC": 
		structure_type = 2
	if Geometry=="HCP":
		structure_type = 3
	if Geometry=="Ico":
		structure_type = 4
	if Geometry=="Dec":
		structure_type = 5
	if Geometry=="Oct":
		structure_type = 6
	return structure_type

def structure_maker(Geometry):
	global h, k, l, atom, lc, lc1, lc2, size1, size2, size3, shell, p, q, r, length
	if Geometry=="SC": 
		stru = SimpleCubic(atom, surfaces=[[1,0,0], [1,1,0], [1,1,1]], layers=(h,k,l), latticeconstant=lc)
		structure_type = 0
	if Geometry=="FCC": 
		stru = FaceCenteredCubic(atom, surfaces=[[1,0,0], [1,1,0], [1,1,1]], layers=(h,k,l), latticeconstant=2*np.sqrt(0.5*lc**2))
		structure_type = 1
	if Geometry=="BCC": 
		stru = BodyCenteredCubic(atom, surfaces=[[1,0,0], [1,1,0], [1,1,1]], layers=(h,k,l), latticeconstant=lc)
		structure_type = 2
	if Geometry=="HCP": 
		stru = HexagonalClosedPacked(symbol=atom, latticeconstant=(lc1, lc2), size=(size1, size2, size3))
		structure_type = 3
	if Geometry=="Ico":
		stru = Icosahedron(atom, shell, latticeconstant=2*np.sqrt(0.5*lc**2))
		structure_type = 4
	if Geometry=="Dec":
		stru = Decahedron(atom, p, q, r, latticeconstant=2*np.sqrt(0.5*lc**2))
		structure_type = 5
	if Geometry=="Oct":
		stru = Octahedron(atom, length, latticeconstant=2*np.sqrt(0.5*lc**2))
		structure_type = 6
	return stru, structure_type

def make_data(atoms, path, numberatoms, Geometry, minimum_atoms=0, numberOfBondLengths=1):
	global ATOM_LIST
	ATOM_LIST = atoms
	if Geometry=="SC" or Geometry=="FCC" or Geometry=="BCC":
		make_data_SC_FCC_BCC(path, numberatoms, Geometry, minimum_atoms, numberOfBondLengths)
	if Geometry=="HCP":
		make_data_HCP(path, numberatoms, Geometry, minimum_atoms, numberOfBondLengths)
	if Geometry=="Ico":
		make_data_Icosahedron(path, numberatoms, Geometry, minimum_atoms, numberOfBondLengths)
	if Geometry=="Dec":
		make_data_Decahedron(path, numberatoms, Geometry, minimum_atoms, numberOfBondLengths)
	if Geometry=="Oct":
		make_data_Octahedron(path, numberatoms, Geometry, minimum_atoms, numberOfBondLengths)
	return None


def make_data_SC_FCC_BCC(path, numberatoms, Geometry, minimum_atoms=0, numberOfBondLengths=1):
	global h, k, l, atom, lc
	if numberatoms < 101:
		h_list = [1, 2, 3, 4, 5, 6, 7, 8]
		k_list = [1, 2, 3, 4, 5, 6, 7]
		l_list = [1, 2, 3, 4, 5, 6, 7]
	elif numberatoms < 201 and numberatoms > 101:
		h_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
		k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
		l_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	elif numberatoms > 201:
		h_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
		k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
		l_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

	possible_structures = len(h_list)*len(k_list)*len(l_list)*len(ATOM_LIST)*numberOfBondLengths
	pbar = tqdm(total=possible_structures)
	for atom in ATOM_LIST:
		structure_list = []
		for h in h_list:
			for k in k_list:
				for l in l_list:	
					if h <= k and h <= l and k <= l:	
						#atom_cov_radii = covalent_radii[atomic_numbers[atom]]
						atom_cov_radii =  mendeleev.element(atom).metallic_radius/100
						lc_list = np.linspace(atom_cov_radii*2*0.99 , atom_cov_radii*2*1.01, numberOfBondLengths)
						if numberOfBondLengths == 1:
							lc_list = [atom_cov_radii*2]
						for lc in lc_list:
							stru1, structure_type = structure_maker(Geometry)
							xyz1 = stru1.get_positions()
							if np.shape(xyz1)[0] > 1 and np.shape(xyz1)[0] <= numberatoms and  np.shape(xyz1)[0] >= minimum_atoms:
								cluster = Structure([Atom(atom, xi) for xi in xyz1])
								if new_structure_checker(xyz1, structure_list) == False:
									structure = np.zeros((numberatoms, 4))
									for i in range(len(xyz1)):
										structure[i, 0] = atomic_numbers[atom]
										structure[i, 1:] = xyz1[i]
									structure_list.append(xyz1)
									cluster.write(path+str(Geometry)+"_h_"+str(h)+"_k_"+str(k)+"_l_"+str(l)+"_atom_"+str(atom)+"_lc_"+str(lc)+".xyz", format="xyz")
									#generator = simPDFs(path+str(Geometry)+"_h_"+str(h)+"_k_"+str(k)+"_l_"+str(l)+"_atom_"+str(atom)+"_lc_"+str(lc)+".xyz", "./")
									#r, Gr = generator.getPDF()
									#y = [structure_type, h, k, l, 0, 0, 0, 0, 0, 0, 0, 0, atomic_numbers[atom], lc, 0]+Gr.tolist()
									#np.savetxt(path+str(Geometry)+"_h_"+str(h)+"_k_"+str(k)+"_l_"+str(l)+"_atom_"+str(atom)+"_lc_"+str(lc)+"_LABEL.txt", y)
					pbar.update(1)
	pbar.close()     
	return None

def make_data_HCP(path, numberatoms, Geometry, minimum_atoms=0, numberOfBondLengths=1):
	global size1, size2, size3, atom, lc1, lc2
	if numberatoms < 101:
		size1_list = [1, 2, 3, 4, 5, 6, 7, 8]
		size2_list = [1, 2, 3, 4, 5, 6, 7, 8]
		size3_list = [1, 2, 3, 4, 5, 6, 7, 8]
	elif numberatoms < 201 and numberatoms > 101:
		size1_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
		size2_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
		size3_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	elif numberatoms > 201:
		size1_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
		size2_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
		size3_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

	possible_structures = len(size1_list)*len(size2_list)*len(size3_list)*len(ATOM_LIST)*numberOfBondLengths
	pbar = tqdm(total=possible_structures)
	for atom in ATOM_LIST:
		structure_list = []
		for size1 in size1_list:
			for size2 in size2_list:
				for size3 in size3_list:
					if size1 <= size2 and size1 <= size3 and size2 <= size3:	
						#atom_cov_radii = covalent_radii[atomic_numbers[atom]]
						atom_cov_radii =  mendeleev.element(atom).metallic_radius/100
						lc1_list = np.linspace(atom_cov_radii*2*0.99 , atom_cov_radii*2*1.01, numberOfBondLengths)
						if numberOfBondLengths == 1:
							lc1_list = [atom_cov_radii*2]
						for lc1 in lc1_list:
							#lc2_list = np.linspace(lc1*2*0.99 , lc1*2*1.01, 10)*1.633
							lc2_list = np.asarray([lc1])*1.633
							for lc2 in lc2_list:
								stru1, structure_type = structure_maker(Geometry)
								xyz1 = stru1.get_positions()
								if np.shape(xyz1)[0] > 1 and np.shape(xyz1)[0] <= numberatoms and  np.shape(xyz1)[0] >= minimum_atoms:
									cluster = Structure([Atom(atom, xi) for xi in xyz1])
									if new_structure_checker(xyz1, structure_list) == False:
										structure = np.zeros((numberatoms, 4))
										for i in range(len(xyz1)):
											structure[i, 0] = atomic_numbers[atom]
											structure[i, 1:] = xyz1[i]
										structure_list.append(xyz1)
										cluster.write(path+str(Geometry)+"_Size1_"+str(size1)+"_Size2_"+str(size2)+"_Size3_"+str(size3)+"_atom_"+str(atom)+"_lc1_"+str(lc1)+"_lc2_"+str(lc2)+".xyz", format="xyz")
										#generator = simPDFs(path+str(Geometry)+"_Size1_"+str(size1)+"_Size2_"+str(size2)+"_Size3_"+str(size3)+"_atom_"+str(atom)+"_lc1_"+str(lc1)+"_lc2_"+str(lc2)+".xyz", "./")
										#r, Gr = generator.getPDF()
										#y = [structure_type, 0, 0, 0, size1, size2, size3, 0, 0, 0, 0, 0, atomic_numbers[atom], lc1, lc2]+Gr.tolist()
										#np.savetxt(path+str(Geometry)+"_Size1_"+str(size1)+"_Size2_"+str(size2)+"_Size3_"+str(size3)+"_atom_"+str(atom)+"_lc1_"+str(lc1)+"_lc2_"+str(lc2)+"_LABEL.txt", y)
								pbar.update(1)
	pbar.close()
	return None

def make_data_Icosahedron(path, numberatoms, Geometry, minimum_atoms=0, numberOfBondLengths=1):
	global atom, shell, lc
	if numberatoms < 101:
		shell_list = [1, 2, 3, 4]
	elif numberatoms < 201 and numberatoms > 101:
		shell_list = [1, 2, 3, 4, 5]
	elif numberatoms > 201:
		shell_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

	possible_structures = len(shell_list)*len(ATOM_LIST)*numberOfBondLengths
	pbar = tqdm(total=possible_structures)
	for atom in ATOM_LIST:
		structure_list = []
		for shell in shell_list:	
			#atom_cov_radii = covalent_radii[atomic_numbers[atom]]
			atom_cov_radii =  mendeleev.element(atom).metallic_radius/100
			lc_list = np.linspace(atom_cov_radii*2*0.99 , atom_cov_radii*2*1.01, numberOfBondLengths)
			if numberOfBondLengths == 1:
				lc_list = [atom_cov_radii*2]
			for lc in lc_list:
				stru1, structure_type = structure_maker(Geometry)
				xyz1 = stru1.get_positions()
				if np.shape(xyz1)[0] > 1 and np.shape(xyz1)[0] <= numberatoms and  np.shape(xyz1)[0] >= minimum_atoms:
					cluster = Structure([Atom(atom, xi) for xi in xyz1])
					if new_structure_checker(xyz1, structure_list) == False:
						structure = np.zeros((numberatoms, 4))
						for i in range(len(xyz1)):
							structure[i, 0] = atomic_numbers[atom]
							structure[i, 1:] = xyz1[i]
						structure_list.append(xyz1)
						cluster.write(path+str(Geometry)+"_shell_"+str(shell)+"_atom_"+str(atom)+"_lc_"+str(lc)+".xyz", format="xyz")
						#generator = simPDFs(path+str(Geometry)+"_shell_"+str(shell)+"_atom_"+str(atom)+"_lc_"+str(lc)+".xyz", "./")
						#r, Gr = generator.getPDF()
						#y = [structure_type, 0, 0, 0, 0, 0, 0, shell, 0, 0, 0, 0, atomic_numbers[atom], lc, 0]+Gr.tolist()
						#np.savetxt(path+str(Geometry)+"_shell_"+str(shell)+"_atom_"+str(atom)+"_lc_"+str(lc)+"_LABEL.txt", y)
				pbar.update(1)
	pbar.close()
	return None

def make_data_Decahedron(path, numberatoms, Geometry, minimum_atoms=0, numberOfBondLengths=1):
	global atom, p, q, r, lc
	if numberatoms < 101:
		p_list = [1, 2, 3, 4]
		q_list = [1, 2, 3, 4]
		r_list = [0, 1, 2, 3, 4]
	elif numberatoms < 201 and numberatoms > 101:
		p_list = [1, 2, 3, 4, 5]
		q_list = [1, 2, 3, 4, 5]
		r_list = [0, 1, 2, 3, 4, 5]
	elif numberatoms > 201:
		p_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
		q_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
		r_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

	possible_structures = len(p_list)*len(q_list)*len(r_list)*len(ATOM_LIST)*numberOfBondLengths
	pbar = tqdm(total=possible_structures)
	for atom in ATOM_LIST:
		structure_list = []
		for p in p_list:
			for q in q_list:
				for r in r_list:	
					#if p <= q and p <= r and q <= r:	
					#atom_cov_radii = covalent_radii[atomic_numbers[atom]]
					atom_cov_radii =  mendeleev.element(atom).metallic_radius/100
					lc_list = np.linspace(atom_cov_radii*2*0.99 , atom_cov_radii*2*1.01, numberOfBondLengths)
					if numberOfBondLengths == 1:
						lc_list = [atom_cov_radii*2]
					for lc in lc_list:
						stru1, structure_type = structure_maker(Geometry)
						xyz1 = stru1.get_positions()
						if np.shape(xyz1)[0] > 1 and np.shape(xyz1)[0] <= numberatoms and  np.shape(xyz1)[0] >= minimum_atoms:
							cluster = Structure([Atom(atom, xi) for xi in xyz1])
							if new_structure_checker(xyz1, structure_list) == False:
								structure = np.zeros((numberatoms, 4))
								for i in range(len(xyz1)):
									structure[i, 0] = atomic_numbers[atom]
									structure[i, 1:] = xyz1[i]
								structure_list.append(xyz1)
								cluster.write(path+str(Geometry)+"_p_"+str(p)+"_q_"+str(q)+"_r_"+str(r)+"_atom_"+str(atom)+"_lc_"+str(lc)+".xyz", format="xyz")
								#generator = simPDFs(path+str(Geometry)+"_p_"+str(p)+"_q_"+str(q)+"_r_"+str(r)+"_atom_"+str(atom)+"_lc_"+str(lc)+".xyz", "./")
								#r_PDF, Gr = generator.getPDF()
								#y = [structure_type, 0, 0, 0, 0, 0, 0, 0, p, q, r, 0, atomic_numbers[atom], lc, 0]+Gr.tolist()
								#np.savetxt(path+str(Geometry)+"_p_"+str(p)+"_q_"+str(q)+"_r_"+str(r)+"_atom_"+str(atom)+"_lc_"+str(lc)+"_LABEL.txt", y)
						pbar.update(1)
	pbar.close()
	return None

def make_data_Octahedron(path, numberatoms, Geometry, minimum_atoms=0, numberOfBondLengths=1):
	global atom, length, lc
	if numberatoms < 101:
		length_list = [2, 3, 4, 5]
	elif numberatoms < 201 and numberatoms > 101:
		length_list = [2, 3, 4, 5, 6, 7]
	elif numberatoms > 201:
		length_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

	possible_structures = len(length_list)*len(ATOM_LIST)*numberOfBondLengths
	pbar = tqdm(total=possible_structures)
	for atom in ATOM_LIST:
		structure_list = []
		for length in length_list:	
			#atom_cov_radii = covalent_radii[atomic_numbers[atom]]
			atom_cov_radii =  mendeleev.element(atom).metallic_radius/100
			lc_list = np.linspace(atom_cov_radii*2*0.99 , atom_cov_radii*2*1.01, numberOfBondLengths)
			if numberOfBondLengths == 1:
				lc_list = [atom_cov_radii*2]
			for lc in lc_list:
				stru1, structure_type = structure_maker(Geometry)
				xyz1 = stru1.get_positions()
				if np.shape(xyz1)[0] > 1 and np.shape(xyz1)[0] <= numberatoms and  np.shape(xyz1)[0] >= minimum_atoms:
					cluster = Structure([Atom(atom, xi) for xi in xyz1])
					if new_structure_checker(xyz1, structure_list) == False:
						structure = np.zeros((numberatoms, 4))
						for i in range(len(xyz1)):
							structure[i, 0] = atomic_numbers[atom]
							structure[i, 1:] = xyz1[i]
						structure_list.append(xyz1)
						cluster.write(path+str(Geometry)+"_length_"+str(length)+"_atom_"+str(atom)+"_lc_"+str(lc)+".xyz", format="xyz")
						#generator = simPDFs(path+str(Geometry)+"_length_"+str(length)+"_atom_"+str(atom)+"_lc_"+str(lc)+".xyz", "./")
						#r, Gr = generator.getPDF()
						#y = [structure_type, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, length, atomic_numbers[atom], lc, 0]+Gr.tolist()
						#np.savetxt(path+str(Geometry)+"_length_"+str(length)+"_atom_"+str(atom)+"_lc_"+str(lc)+"_LABEL.txt", y)
				pbar.update(1)
	pbar.close()
	return None



def new_structure_checker(Input_array, list_search):
	for i in range(len(list_search)):
		if np.all(Input_array == list_search[i]) == True:
			return True
		else:
			pass
	return False



def gen_xyz(atoms, type_list, max_atoms, interpolate, directory_base):
	if directory_base==None:
		ct = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '-')
		directory_base = f'./data_{ct}'
		print(f'\nProject name is: {directory_base}')
	elif directory_base[-1] != '/':
		directory_base = f'{directory_base}/'

	directory = f"{directory_base}/xyz_raw/"
	try:
		os.makedirs(directory)
	except FileExistsError:
		pass

	for type in type_list:
		print(f'Generating structure type: {type}')
		make_data(atoms, directory, max_atoms, type, minimum_atoms=5, numberOfBondLengths=interpolate)


	return directory_base
