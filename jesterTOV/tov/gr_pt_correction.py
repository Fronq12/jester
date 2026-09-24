r"""
General Relativity TOV equation solver.

This module implements the standard Tolman-Oppenheimer-Volkoff equations
for hydrostatic equilibrium in General Relativity.

**Units:** All calculations are performed in geometric units where :math:`G = c = 1`.

**Reference:** NMMA code https://github.com/nuclear-multimessenger-astronomy/nmma/
"""

import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController

from jesterTOV import utils
from jesterTOV.tov.base import TOVSolverBase
from jesterTOV.tov.data_classes import EOSData, TOVSolution


def _tov_ode(h, y, eos):
    r"""
    TOV ordinary differential equation system.

    This function defines the coupled ODE system for the TOV equations plus
    tidal deformability. The system is solved in terms of the enthalpy h as
    the independent variable (decreasing from center to surface).

    The TOV equations are:

    .. math::
        \frac{dr}{dh} &= -\frac{r(r-2m)}{m + 4\pi r^3 p} \\
        \frac{dm}{dh} &= 4\pi r^2 \varepsilon \frac{dr}{dh} \\
        \frac{dH}{dh} &= \beta \frac{dr}{dh} \\
        \frac{d\beta}{dh} &= -(C_0 H + C_1 \beta) \frac{dr}{dh}

    where H and :math:`\beta` are auxiliary variables for tidal deformability.

    Parameters
    ----------
    h : float
        Enthalpy (independent variable)
    y : tuple
        State vector (r, m, H, β)
    eos : dict
        EOS interpolation data

    Returns
    -------
    tuple
        Derivatives (dr/dh, dm/dh, dH/dh, dβ/dh)
    """
    # Extract EOS interpolation arrays
    ps = eos["p"]
    hs = eos["h"]
    es = eos["e"]
    dloge_dlogps = eos["dloge_dlogp"]
    # Extract current state variables
    r, m, H, b = y

    # Guard h against non-positive values.  Diffrax's adaptive step controller
    # evaluates the ODE at tentative points that can overshoot t1=0 into h <= 0.
    # interp_in_logspace computes jnp.log(h), which is NaN for h < 0, producing
    # a NaN that propagates into the adjoint backward pass and corrupts gradients.
    # Clamping h to the bottom of the EOS table freezes the ODE derivatives at
    # their surface value for out-of-range enthalpies, which is physically correct
    # and leaves all forward-pass outputs (M, R, k2) bit-for-bit unchanged.
    h_safe = jnp.maximum(h, hs[0])

    e = utils.interp_in_logspace(h_safe, hs, es)
    p = utils.interp_in_logspace(h_safe, hs, ps)
    dedp = e / p * jnp.interp(h_safe, hs, dloge_dlogps)

    # Metric coefficient A = 1/(1-2m/r)
    A = 1.0 / (1.0 - 2.0 * m / r)
    # Tidal deformability coefficients
    C1 = 2.0 / r + A * (2.0 * m / (r * r) + 4.0 * jnp.pi * r * (p - e))
    C0 = A * (
        -6 / (r * r)
        + 4.0 * jnp.pi * (e + p) * dedp
        + 4.0 * jnp.pi * (5.0 * e + 9.0 * p)
    ) - jnp.power(2.0 * (m + 4.0 * jnp.pi * r * r * r * p) / (r * (r - 2.0 * m)), 2.0)

    # TOV equation derivatives
    drdh = -r * (r - 2.0 * m) / (m + 4.0 * jnp.pi * r * r * r * p)  # dr/dh
    dmdh = 4.0 * jnp.pi * r * r * e * drdh  # dm/dh
    dHdh = b * drdh  # dH/dh
    dbdh = -(C0 * H + C1 * b) * drdh  # dβ/dh

    return drdh, dmdh, dHdh, dbdh


def _calc_k2(R, M, H, b, correction_term):
    r"""
    Calculate the second Love number k₂ for tidal deformability.

    The Love number k₂ relates the tidal deformability to the neutron star's
    mass and radius. It is computed from the auxiliary variables H and β
    obtained from the TOV integration.

    The tidal deformability is given by:

    .. math::
        \Lambda = \frac{2}{3} k_2 C^{-5}

    where :math:`C = M/R` is the compactness.

    Parameters
    ----------
    R : float
        Neutron star radius [geometric units]
    M : float
        Neutron star mass [geometric units]
    H : float
        Auxiliary tidal variable at surface
    b : float
        Auxiliary tidal variable β at surface

    Returns
    -------
    float
        Second Love number k₂
    """
    y = R * b / H - correction_term # The correction term is zero if there whas no correction
    C = M / R

    num = (
        (8.0 / 5.0)
        * jnp.power(1 - 2 * C, 2.0)
        * jnp.power(C, 5.0)
        * (2 * C * (y - 1) - y + 2)
    )
    den = (
        2
        * C
        * (
            4 * (y + 1) * jnp.power(C, 4)
            + (6 * y - 4) * jnp.power(C, 3)
            + (26 - 22 * y) * C * C
            + 3 * (5 * y - 8) * C
            - 3 * y
            + 6
        )
    )
    den -= (
        3
        * jnp.power(1 - 2 * C, 2)
        * (2 * C * (y - 1) - y + 2)
        * jnp.log(1.0 / (1 - 2 * C))
    )

    return num / den

# Including correction due to first order PT according to eq.(14) of Sergey Postnikov and Madappa Prakash, 2010 arXiv:1004.5098v1
def correction_phase_transition(R, M, dt_e):
    r"""
    press_pt: Phase transition pressure
    dt_e: Energy density gap of the PT
    """
    # Requires the energy jump during phase transition, do not know how to get this yet 
    # Requires radius at discontinuity and the corresponding mass, do not know how to get this yet 
    # We need a method to identify the phase transition pressure 

    num = 4 * jnp.pi * dt_e * jnp.power(R,3) # We should only save the correction term 
    den = M
    return num / den 

def find_phase_transition(ps, hs, es ): 
    """
    Function that finds phase transition locations.

    Finds an array whit the starting indices and an array whit the ending indices. 
    Returns the values of pressure and enthalpy at the starting points and the energy upper and lower 
    boundaries around phase transition

    ps: array of pressures from the eos
    hs: array of enthalpies from the eos
    es: array of energy densities from the eos

    Might need updateing depending on how the EOS is generated 
    """
    # We define an array that contains Trues for each index where at least one neighbor has the same pressure value 
    PT_mask = ps[1:] == ps[:-1] # TODO might need to give some range of how much we allow the pressure to differ

    difference = jnp.diff(PT_mask.astype(int), prepend= 0) #all startting points will be 1 and all end points will be -1

    # Note start and end are defined going from low to high pressure 
    start_idx = jnp.where(difference == 1)[0] # find and save all indices of the PT starting locations 
    end_idx = jnp.where(difference == -1)[0] # find and save all indices of the PT end locations

    # Find the values for the pressure and enthalpy starting points 
    pt_p = ps[start_idx]
    pt_h = hs[start_idx]

    # Find the energy upper and lower boundary 
    pt_es_low = es[start_idx]
    pt_es_high = es[end_idx]

    # Calculate energy jumps at each PT
    dt_e = pt_es_high - pt_es_low

    # we return a dictionary containing all relevant PT information 
    boundary_dict = {
        "p": pt_p,
        "h": pt_h,
        "e_low": pt_es_low,
        "e_high": pt_es_high,
        "delta_e": dt_e,
        "start_idx": start_idx,
        "end_idx": end_idx
    }
    return boundary_dict

class GRTOVSolver(TOVSolverBase):
    """
    Standard General Relativity TOV solver.

    Solves the TOV equations:

    .. math::
        \\frac{dr}{dh} &= -\\frac{r(r-2m)}{m + 4\\pi r^3 p} \\\\
        \\frac{dm}{dh} &= 4\\pi r^2 \\varepsilon \\frac{dr}{dh}

    plus the equations for tidal deformability.
    """

    def solve(
        self, eos_data: EOSData, pc: float, tov_params: dict[str, float] = {}
    ) -> TOVSolution:
        r"""
        Solve TOV equations for given central pressure.

        This function integrates the TOV equations from the center of the star
        (where r=0, m=0) outward to the surface (where p=0), using the enthalpy
        as the integration variable. The integration starts slightly off-center
        to avoid singularities.

        The solver uses the Dormand-Prince 5th order adaptive method (Dopri5)
        with proper error control for numerical stability.

        Args:
            eos_data: EOS quantities in geometric units
            pc: Central pressure [geometric units]
            tov_params: Not used for GR TOV (defaults to ``{}`` — no additional parameters)

        Returns:
            TOVSolution: Mass, radius, and Love number in geometric units.
                        Returns NaN values on solver failure (JAX-compatible).

        Notes:
            The integration is performed from center to surface, with the enthalpy
            decreasing from h_center to 0. Initial conditions are set using
            series expansions valid near the center.

        Notes on phase transitions:
            Everytime that we encounter a PT we will have to break the integration and calculate M,R,H,b and k2
            We then skip over the phase transition to the next continuous piece of EOS and repeat 
        """
        # Convert EOSData to dictionary for ODE solver 
        # TODO make this into an EOS pieces dict
        # eos_dict = {
        #     "p": eos_data.ps,
        #     "h": eos_data.hs,
        #     "e": eos_data.es,
        #     "dloge_dlogp": eos_data.dloge_dlogps,
        # }

        # TODO Q: what is the form of these arrays; are they aranged from low to high pressure?
        # Extract EOS arrays
        ps = eos_data.ps # pressure
        hs = eos_data.hs # pseudo enthalpy
        es = eos_data.es # energy density
        dloge_dlogps = eos_data.dloge_dlogps

        #Get the boundaries of the phase transitions 
        eos_piece_boundaries = find_phase_transition(ps, hs, es)
        pt_p = eos_piece_boundaries["p"]

        # Note start and end are defined going from low to high pressure 
        start_idx =eos_piece_boundaries["start_idx"]
        end_idx =eos_piece_boundaries["end_idx"]

        # TODO plase the following in the PT finder function
        # Add the index zero to the start and end array to deal whit a central pressure below PT pressure 
        start_idx = jnp.concatenate(jnp.array([0]),start_idx, jnp.array([len(ps)])) # Add index of last pressure to deal whit boundary
        end_idx = jnp.concatenate(jnp.array([0]),end_idx)
        pt_p = jnp.concatenate(jnp.array([0]), pt_p) # add zero pressure to account for the first piece between the crust and the first PT

        # TODO don't put this in loops!!
        # Find where the central pressure lies whitin the EOS pieces
        central_piece = 0
        for i in range(len(pt_p)):
            pt_pi = pt_p[i]
            if pc > pt_pi:
                central_piece = i #last PT pressure that whas exeeded 

        # Find the ranges of the center piece
        central_low = end_idx[central_piece]
        central_high = start_idx[central_piece + 1]
        central_p_piece = ps[central_low:central_high]
        central_e_piece = es[central_low:central_high]
        central_h_piece = hs[central_low:central_high]
        central_dloge_dlogps = dloge_dlogps[central_low:central_high]

        # TODO write the initial conditions using the correct eos pieces 
        # Central values and initial conditions
        hc = utils.interp_in_logspace(pc, central_p_piece, central_h_piece) #interpolate to get full range 
        ec = utils.interp_in_logspace(hc, central_h_piece, central_e_piece)
        dedp_c = ec / pc * jnp.interp(hc, central_h_piece, central_dloge_dlogps)
        dhdp_c = 1.0 / (ec + pc)
        dedh_c = dedp_c / dhdp_c

        # Initial values using series expansion near center
        dh = -1e-3 * hc
        h0 = hc + dh
        # h1 = hs[central_piece] #This is h1 after the first run
        r0 = jnp.sqrt(3.0 * (-dh) / 2.0 / jnp.pi / (ec + 3.0 * pc))
        r0 *= 1.0 - 0.25 * (ec - 3.0 * pc - 0.6 * dedh_c) * (-dh) / (ec + 3.0 * pc)
        m0 = 4.0 * jnp.pi * ec * jnp.power(r0, 3.0) / 3.0
        m0 *= 1.0 - 0.6 * dedh_c * (-dh) / ec
        H0 = r0 * r0
        b0 = 2.0 * r0

        y0 = (r0, m0, H0, b0) 

        # init the correction term, remains zero if there is no phase transition
        correctrion_term = 0

        # TODO get rid of the while loop!!!
        # TODO this assumes the eos starts at zero pressure, energy density and enthalpy, need to check this 
        # Perform integration up to last phase transition (closest to the crust)
        dt_e = jnp.concatenate(jnp.array([0]), eos_piece_boundaries["delta_e"])
        while central_piece >= 0:

            # Define argument dictionairy
            c_low = end_idx[central_piece]
            c_high = start_idx[central_piece + 1] # this will not work if the PT is in the highest pressure piece 
            eos_dict = {
                "p": ps[c_low:c_high],
                "h": hs[c_low:c_high],
                "e": es[c_low:c_high],
                "dloge_dlogp": dloge_dlogps[c_low:c_high],
            }
            h1 = hs[c_low]
            sol = diffeqsolve(
                ODETerm(_tov_ode),
                Dopri5(scan_kind="bounded"),
                t0=h0,
                t1=h1,
                dt0=dh,
                y0=y0,
                args=eos_dict,
                saveat=SaveAt(t1=True),
                stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
                throw=False,
            )
            # Calculate 
            R = sol.ys[0][-1] 
            M = sol.ys[1][-1]  
            dt_e = dt_e[central_piece]
            correctrion_term = correction_phase_transition(R,M,dt_e) # Zero if delta e = 0 

            h0 = h1 + -1e-3 *h1 # get new starting position 
            
            central_piece -= 1


        # # TODO Run one more integration to get to the crust 
        # sol = diffeqsolve(
        #     ODETerm(_tov_ode),
        #     Dopri5(scan_kind="bounded"),
        #     t0=h0,
        #     t1=0,
        #     dt0=dh,
        #     y0=y0,
        #     args=eos_dict,
        #     saveat=SaveAt(t1=True),
        #     stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
        #     throw=False,
        # )

        # Extract solution values
        # Note: diffrax always returns arrays with throw=False
        R = sol.ys[0][-1]  # type: ignore[index]
        M = sol.ys[1][-1]  # type: ignore[index]
        H = sol.ys[2][-1]  # type: ignore[index]
        b = sol.ys[3][-1]  # type: ignore[index] # should probably only correct b at the interface

        k2 = _calc_k2(R, M, H, b, correctrion_term)

        return TOVSolution(M=M, R=R, k2=k2)

    def get_required_parameters(self) -> list[str]:
        """
        GR TOV requires no additional parameters beyond EOS.

        Returns:
            list[str]: Empty list (no extra parameters)
        """
        return []
