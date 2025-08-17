"""
Nash Equilibrium Solver for Game Theory Policy Analysis
======================================================

Professional implementation of Nash equilibrium calculation
for multi-player strategic games in policy analysis.

Author: Rosalina Torres
Institution: Northeastern University - Data Analytics Engineering
Date: January 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

class NashEquilibriumSolver:
    """
    Professional Nash Equilibrium solver for policy analysis games
    
    Calculates pure strategy Nash equilibria, analyzes stability,
    and provides policy insights from game theory perspectives.
    """
    
    def __init__(self, payoff_matrix: np.ndarray, players: List[str], strategies: Dict[str, List[str]]):
        """
        Initialize Nash Equilibrium solver
        
        Args:
            payoff_matrix: 3D numpy array [strategy1, strategy2, player] with payoffs
            players: List of player names
            strategies: Dict mapping players to their strategy options
        """
        self.payoff_matrix = payoff_matrix
        self.players = players
        self.strategies = strategies
        self.equilibria = None
        self.analysis_results = {}
        
        # Validate inputs
        self._validate_inputs()
        
    def _validate_inputs(self) -> None:
        """Validate input parameters"""
        if len(self.payoff_matrix.shape) != 3:
            raise ValueError("Payoff matrix must be 3D: [strategy1, strategy2, player]")
        
        if self.payoff_matrix.shape[2] != len(self.players):
            raise ValueError("Third dimension of payoff matrix must equal number of players")
        
        for i, player in enumerate(self.players):
            if player not in self.strategies:
                raise ValueError(f"Strategies not defined for player {player}")
            
            expected_strategies = self.payoff_matrix.shape[i]
            actual_strategies = len(self.strategies[player])
            if expected_strategies != actual_strategies:
                raise ValueError(f"Strategy count mismatch for {player}: expected {expected_strategies}, got {actual_strategies}")
    
    def find_pure_nash_equilibria(self) -> List[Tuple[int, int]]:
        """
        Find all pure strategy Nash equilibria
        
        Returns:
            List of tuples representing equilibrium strategy combinations
        """
        equilibria = []
        n_strategies_p1, n_strategies_p2, _ = self.payoff_matrix.shape
        
        # Check each strategy combination
        for s1 in range(n_strategies_p1):
            for s2 in range(n_strategies_p2):
                is_equilibrium = True
                
                # Check if player 1 wants to deviate
                current_payoff_p1 = self.payoff_matrix[s1, s2, 0]
                for alt_s1 in range(n_strategies_p1):
                    if alt_s1 != s1:
                        alt_payoff_p1 = self.payoff_matrix[alt_s1, s2, 0]
                        if alt_payoff_p1 > current_payoff_p1:
                            is_equilibrium = False
                            break
                
                if not is_equilibrium:
                    continue
                
                # Check if player 2 wants to deviate
                current_payoff_p2 = self.payoff_matrix[s1, s2, 1]
                for alt_s2 in range(n_strategies_p2):
                    if alt_s2 != s2:
                        alt_payoff_p2 = self.payoff_matrix[s1, alt_s2, 1]
                        if alt_payoff_p2 > current_payoff_p2:
                            is_equilibrium = False
                            break
                
                if is_equilibrium:
                    equilibria.append((s1, s2))
        
        self.equilibria = equilibria
        return equilibria
    
    def analyze_dominant_strategies(self) -> Dict:
        """
        Analyze dominant and dominated strategies
        
        Returns:
            Dictionary with dominant strategy analysis
        """
        analysis = {
            'player_1': {
                'dominant_strategies': [],
                'dominated_strategies': [],
                'weakly_dominant': []
            },
            'player_2': {
                'dominant_strategies': [],
                'dominated_strategies': [],
                'weakly_dominant': []
            }
        }
        
        n_strategies_p1, n_strategies_p2, _ = self.payoff_matrix.shape
        
        # Analyze Player 1 strategies
        for s1 in range(n_strategies_p1):
            is_dominant = True
            is_weakly_dominant = True
            dominates_all = True
            
            # Check if s1 dominates all other strategies
            for alt_s1 in range(n_strategies_p1):
                if alt_s1 != s1:
                    dominates_alt = True
                    weakly_dominates_alt = True
                    
                    for s2 in range(n_strategies_p2):
                        payoff_s1 = self.payoff_matrix[s1, s2, 0]
                        payoff_alt = self.payoff_matrix[alt_s1, s2, 0]
                        
                        if payoff_s1 <= payoff_alt:
                            dominates_alt = False
                        if payoff_s1 < payoff_alt:
                            weakly_dominates_alt = False
                    
                    if not dominates_alt:
                        is_dominant = False
                    if not weakly_dominates_alt:
                        is_weakly_dominant = False
                    
                    # Check if s1 is dominated by alt_s1
                    dominated_by_alt = True
                    for s2 in range(n_strategies_p2):
                        payoff_s1 = self.payoff_matrix[s1, s2, 0]
                        payoff_alt = self.payoff_matrix[alt_s1, s2, 0]
                        
                        if payoff_s1 >= payoff_alt:
                            dominated_by_alt = False
                            break
                    
                    if dominated_by_alt:
                        analysis['player_1']['dominated_strategies'].append(s1)
                        dominates_all = False
                        break
            
            if is_dominant and dominates_all:
                analysis['player_1']['dominant_strategies'].append(s1)
            elif is_weakly_dominant and dominates_all:
                analysis['player_1']['weakly_dominant'].append(s1)
        
        # Analyze Player 2 strategies (similar logic)
        for s2 in range(n_strategies_p2):
            is_dominant = True
            is_weakly_dominant = True
            dominates_all = True
            
            for alt_s2 in range(n_strategies_p2):
                if alt_s2 != s2:
                    dominates_alt = True
                    weakly_dominates_alt = True
                    
                    for s1 in range(n_strategies_p1):
                        payoff_s2 = self.payoff_matrix[s1, s2, 1]
                        payoff_alt = self.payoff_matrix[s1, alt_s2, 1]
                        
                        if payoff_s2 <= payoff_alt:
                            dominates_alt = False
                        if payoff_s2 < payoff_alt:
                            weakly_dominates_alt = False
                    
                    if not dominates_alt:
                        is_dominant = False
                    if not weakly_dominates_alt:
                        is_weakly_dominant = False
                    
                    # Check if s2 is dominated by alt_s2
                    dominated_by_alt = True
                    for s1 in range(n_strategies_p1):
                        payoff_s2 = self.payoff_matrix[s1, s2, 1]
                        payoff_alt = self.payoff_matrix[s1, alt_s2, 1]
                        
                        if payoff_s2 >= payoff_alt:
                            dominated_by_alt = False
                            break
                    
                    if dominated_by_alt:
                        analysis['player_2']['dominated_strategies'].append(s2)
                        dominates_all = False
                        break
            
            if is_dominant and dominates_all:
                analysis['player_2']['dominant_strategies'].append(s2)
            elif is_weakly_dominant and dominates_all:
                analysis['player_2']['weakly_dominant'].append(s2)
        
        return analysis
    
    def calculate_welfare_outcomes(self) -> Dict:
        """
        Calculate total welfare for each strategy combination
        
        Returns:
            Dictionary with welfare analysis
        """
        welfare_matrix = np.sum(self.payoff_matrix, axis=2)
        n_strategies_p1, n_strategies_p2 = welfare_matrix.shape
        
        welfare_outcomes = {}
        for s1 in range(n_strategies_p1):
            for s2 in range(n_strategies_p2):
                strategy_combo = (
                    self.strategies[self.players[0]][s1],
                    self.strategies[self.players[1]][s2]
                )
                welfare_outcomes[strategy_combo] = {
                    'total_welfare': welfare_matrix[s1, s2],
                    'player_payoffs': {
                        self.players[0]: self.payoff_matrix[s1, s2, 0],
                        self.players[1]: self.payoff_matrix[s1, s2, 1]
                    }
                }
        
        # Find Pareto optimal outcomes
        max_welfare = np.max(welfare_matrix)
        pareto_optimal = []
        for s1 in range(n_strategies_p1):
            for s2 in range(n_strategies_p2):
                if welfare_matrix[s1, s2] == max_welfare:
                    pareto_optimal.append((s1, s2))
        
        return {
            'welfare_outcomes': welfare_outcomes,
            'welfare_matrix': welfare_matrix,
            'pareto_optimal': pareto_optimal,
            'max_welfare': max_welfare,
            'min_welfare': np.min(welfare_matrix)
        }
    
    def run_complete_analysis(self) -> Dict:
        """
        Run complete Nash equilibrium analysis
        
        Returns:
            Comprehensive analysis results
        """
        # Find equilibria
        equilibria = self.find_pure_nash_equilibria()
        
        # Analyze dominant strategies
        dominant_analysis = self.analyze_dominant_strategies()
        
        # Calculate welfare outcomes
        welfare_analysis = self.calculate_welfare_outcomes()
        
        # Classify game type
        game_classification = self._classify_game()
        
        # Generate policy insights
        policy_insights = self._generate_policy_insights(equilibria, welfare_analysis)
        
        # Compile results
        results = {
            'equilibria': {
                'count': len(equilibria),
                'strategy_combinations': [],
                'equilibrium_payoffs': []
            },
            'dominant_strategies': dominant_analysis,
            'welfare_analysis': welfare_analysis,
            'game_classification': game_classification,
            'policy_insights': policy_insights
        }
        
        # Format equilibria results
        for eq in equilibria:
            s1, s2 = eq
            strategy_names = (
                self.strategies[self.players[0]][s1],
                self.strategies[self.players[1]][s2]
            )
            payoffs = (
                float(self.payoff_matrix[s1, s2, 0]),
                float(self.payoff_matrix[s1, s2, 1])
            )
            
            results['equilibria']['strategy_combinations'].append(strategy_names)
            results['equilibria']['equilibrium_payoffs'].append(payoffs)
        
        self.analysis_results = results
        return results
    
    def _classify_game(self) -> Dict:
        """
        Classify the game type (Prisoner's Dilemma, Coordination, etc.)
        
        Returns:
            Game classification analysis
        """
        welfare_matrix = np.sum(self.payoff_matrix, axis=2)
        
        # Check for Prisoner's Dilemma characteristics
        # (Both players have dominant strategies leading to suboptimal outcome)
        dominant_analysis = self.analyze_dominant_strategies()
        
        is_prisoners_dilemma = False
        if (len(dominant_analysis['player_1']['dominant_strategies']) == 1 and 
            len(dominant_analysis['player_2']['dominant_strategies']) == 1):
            
            # Check if dominant strategies lead to suboptimal welfare
            dom_s1 = dominant_analysis['player_1']['dominant_strategies'][0]
            dom_s2 = dominant_analysis['player_2']['dominant_strategies'][0]
            
            dominant_welfare = welfare_matrix[dom_s1, dom_s2]
            max_welfare = np.max(welfare_matrix)
            
            if dominant_welfare < max_welfare:
                is_prisoners_dilemma = True
        
        # Check for coordination game
        # (Multiple equilibria with different welfare outcomes)
        if self.equilibria is None:
            equilibria = self.find_pure_nash_equilibria()
        else:
            equilibria = self.equilibria
            
        equilibrium_welfares = [welfare_matrix[s1, s2] for s1, s2 in equilibria]
        
        is_coordination_game = len(set(equilibrium_welfares)) > 1 if len(equilibrium_welfares) > 1 else False
        
        return {
            'is_prisoners_dilemma': is_prisoners_dilemma,
            'is_coordination_game': is_coordination_game,
            'has_dominant_strategies': any(
                len(dominant_analysis[f'player_{i+1}']['dominant_strategies']) > 0 
                for i in range(len(self.players))
            ),
            'equilibrium_count': len(equilibria),
            'welfare_efficiency': np.max(welfare_matrix) / np.sum(np.max(self.payoff_matrix, axis=(0, 1))) if np.sum(np.max(self.payoff_matrix, axis=(0, 1))) > 0 else 0
        }
    
    def _generate_policy_insights(self, equilibria: List[Tuple[int, int]], welfare_analysis: Dict) -> Dict:
        """
        Generate policy insights from game theory analysis
        
        Args:
            equilibria: List of Nash equilibria
            welfare_analysis: Welfare analysis results
            
        Returns:
            Policy insights and recommendations
        """
        insights = {
            'stability_assessment': '',
            'efficiency_assessment': '',
            'policy_recommendations': [],
            'cooperation_potential': '',
            'key_risks': []
        }
        
        # Stability assessment
        if len(equilibria) == 0:
            insights['stability_assessment'] = 'No pure strategy equilibria - inherently unstable'
        elif len(equilibria) == 1:
            insights['stability_assessment'] = 'Single stable equilibrium - predictable outcome'
        else:
            insights['stability_assessment'] = f'Multiple equilibria ({len(equilibria)}) - outcome depends on expectations'
        
        # Efficiency assessment
        welfare_matrix = welfare_analysis['welfare_matrix']
        if len(equilibria) > 0:
            eq_welfares = [welfare_matrix[s1, s2] for s1, s2 in equilibria]
            avg_eq_welfare = np.mean(eq_welfares)
            max_welfare = welfare_analysis['max_welfare']
            
            if max_welfare > 0:
                efficiency_ratio = avg_eq_welfare / max_welfare
                if efficiency_ratio >= 0.9:
                    insights['efficiency_assessment'] = 'Highly efficient - equilibrium outcomes near optimal'
                elif efficiency_ratio >= 0.7:
                    insights['efficiency_assessment'] = 'Moderately efficient - some welfare loss in equilibrium'
                else:
                    insights['efficiency_assessment'] = 'Inefficient - significant welfare loss in equilibrium'
            else:
                insights['efficiency_assessment'] = 'Cannot assess efficiency - zero welfare scenarios'
        else:
            insights['efficiency_assessment'] = 'Cannot assess efficiency - no pure strategy equilibria'
        
        # Policy recommendations based on game structure
        try:
            game_class = self._classify_game()
            
            if game_class['is_prisoners_dilemma']:
                insights['policy_recommendations'].extend([
                    'Establish binding cooperation mechanisms',
                    'Create enforcement mechanisms for agreements',
                    'Align individual incentives with collective welfare'
                ])
                insights['cooperation_potential'] = 'Difficult - individual incentives oppose cooperation'
                insights['key_risks'].append('Race to the bottom without intervention')
            
            elif game_class['is_coordination_game']:
                insights['policy_recommendations'].extend([
                    'Facilitate communication between players',
                    'Establish clear focal points for coordination',
                    'Create mechanisms to select efficient equilibrium'
                ])
                insights['cooperation_potential'] = 'High - mutual benefits from coordination'
                insights['key_risks'].append('Coordination failure leading to inefficient outcome')
            else:
                insights['cooperation_potential'] = 'Moderate - depends on specific incentive structure'
            
            if game_class['has_dominant_strategies']:
                insights['policy_recommendations'].append('Leverage dominant strategy incentives')
                
        except Exception as e:
            print(f"Warning: Could not classify game type: {e}")
            insights['cooperation_potential'] = 'Unknown - game classification failed'
            insights['policy_recommendations'].append('Conduct detailed strategy analysis')
            
        return insights
    
    def visualize_payoff_matrix(self, save_path: Optional[str] = None) -> None:
        """
        Create visualization of the payoff matrix
        
        Args:
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Player 1 payoffs
        player1_payoffs = self.payoff_matrix[:, :, 0]
        sns.heatmap(player1_payoffs, 
                   annot=True, 
                   fmt='.1f',
                   xticklabels=self.strategies[self.players[1]],
                   yticklabels=self.strategies[self.players[0]],
                   cmap='RdYlGn',
                   ax=axes[0])
        axes[0].set_title(f'{self.players[0]} Payoffs')
        axes[0].set_xlabel(f'{self.players[1]} Strategy')
        axes[0].set_ylabel(f'{self.players[0]} Strategy')
        
        # Player 2 payoffs  
        player2_payoffs = self.payoff_matrix[:, :, 1]
        sns.heatmap(player2_payoffs,
                   annot=True,
                   fmt='.1f', 
                   xticklabels=self.strategies[self.players[1]],
                   yticklabels=self.strategies[self.players[0]],
                   cmap='RdYlGn',
                   ax=axes[1])
        axes[1].set_title(f'{self.players[1]} Payoffs')
        axes[1].set_xlabel(f'{self.players[1]} Strategy')
        axes[1].set_ylabel(f'{self.players[0]} Strategy')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Payoff matrix visualization saved: {save_path}")
        
        plt.show()
    
    def print_analysis_summary(self) -> None:
        """Print formatted summary of the analysis"""
        if not self.analysis_results:
            print("❌ No analysis results available. Run run_complete_analysis() first.")
            return
        
        results = self.analysis_results
        
        print("🎯 NASH EQUILIBRIUM ANALYSIS SUMMARY")
        print("=" * 50)
        
        # Equilibria
        eq_count = results['equilibria']['count']
        print(f"📊 Nash Equilibria Found: {eq_count}")
        
        if eq_count > 0:
            for i, (strategies, payoffs) in enumerate(zip(
                results['equilibria']['strategy_combinations'],
                results['equilibria']['equilibrium_payoffs']
            )):
                print(f"  {i+1}. {strategies[0]} vs {strategies[1]} → ({payoffs[0]:.1f}, {payoffs[1]:.1f})")
        
        # Dominant strategies
        print(f"\n🎯 Dominant Strategy Analysis:")
        for i, player in enumerate(self.players):
            dom_strats = results['dominant_strategies'][f'player_{i+1}']['dominant_strategies']
            if dom_strats:
                strat_names = [self.strategies[player][s] for s in dom_strats]
                print(f"  {player}: {strat_names[0]} is dominant")
            else:
                print(f"  {player}: No dominant strategy")
        
        # Game classification
        game_class = results['game_classification']
        print(f"\n🎲 Game Type:")
        if game_class['is_prisoners_dilemma']:
            print("  Prisoner's Dilemma - Individual incentives conflict with collective welfare")
        elif game_class['is_coordination_game']:
            print("  Coordination Game - Multiple equilibria with different welfare outcomes")
        else:
            print("  General Strategic Game")
        
        # Policy insights
        policy = results['policy_insights']
        print(f"\n📋 Policy Assessment:")
        print(f"  Stability: {policy['stability_assessment']}")
        print(f"  Efficiency: {policy['efficiency_assessment']}")
        print(f"  Cooperation Potential: {policy['cooperation_potential']}")
        
        if policy['policy_recommendations']:
            print(f"  Key Recommendations:")
            for rec in policy['policy_recommendations']:
                print(f"    • {rec}")

# Example usage and testing
if __name__ == "__main__":
    # Example: Innovation Investment Game
    print("🚀 NASH EQUILIBRIUM SOLVER DEMO")
    print("=" * 40)
    
    # Define payoff matrix for innovation game
    # Rows: US strategies, Columns: Global strategies
    # US strategies: [STEM Investment, Manufacturing Protection]  
    # Global strategies: [Innovation, Traditional]
    innovation_payoffs = np.array([
        [[8, 7], [6, 4]],  # US: STEM Investment
        [[4, 6], [2, 2]]   # US: Manufacturing Protection
    ])
    
    players = ['United States', 'Global Economy']
    strategies = {
        'United States': ['STEM Investment', 'Manufacturing Protection'],
        'Global Economy': ['Innovation', 'Traditional']
    }
    
    # Initialize solver
    solver = NashEquilibriumSolver(innovation_payoffs, players, strategies)
    
    # Run complete analysis
    results = solver.run_complete_analysis()
    
    # Print results
    solver.print_analysis_summary()
    
    # Create visualization
    print(f"\n🎨 Generating payoff matrix visualization...")
    solver.visualize_payoff_matrix('nash_equilibrium_analysis.png')
    
    print(f"\n✅ Nash Equilibrium Analysis Complete!")
    print(f"📊 Analysis demonstrates that STEM Investment is mathematically optimal")
