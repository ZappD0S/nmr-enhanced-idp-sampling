from abc import ABC, abstractmethod


class BaseDataBuilder(ABC):
    @property
    @abstractmethod
    def n_atoms(self):
        pass

    @property
    @abstractmethod
    def n_bonds(self):
        pass

    @property
    @abstractmethod
    def n_angles(self):
        pass

    @property
    @abstractmethod
    def n_dihedrals(self):
        pass

    @property
    @abstractmethod
    def n_crossterms(self):
        pass

    @property
    @abstractmethod
    def n_atom_types(self):
        pass

    @property
    @abstractmethod
    def n_bond_types(self):
        pass

    @property
    @abstractmethod
    def n_angle_types(self):
        pass

    @property
    @abstractmethod
    def n_dihedral_types(self):
        pass

    @property
    @abstractmethod
    def crossterm_type_to_resids(self):
        pass

    @abstractmethod
    def build_atom_type_masses(self):
        pass

    @abstractmethod
    def build_atoms_list(self, ag):
        pass

    @abstractmethod
    def build_pair_coeffs(self):
        pass

    @abstractmethod
    def build_bond_coeffs(self):
        pass

    @abstractmethod
    def build_bonds_list(self):
        pass

    @abstractmethod
    def build_angle_coeffs(self):
        pass

    @abstractmethod
    def build_angles_list(self):
        pass

    @abstractmethod
    def build_dihedral_coeffs(self):
        pass

    @abstractmethod
    def build_dihedrals_list(self):
        pass

    @abstractmethod
    def build_cmap_crossterms_list(self):
        pass
