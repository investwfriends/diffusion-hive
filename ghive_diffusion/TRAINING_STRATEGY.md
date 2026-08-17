### 1. Data Requirements                                            
 ┃                                                                       
 ┃   The larger model (ghive_diffusion) features a Mixture of Experts    
 ┃   (MoE) routing layer, 9 auxiliary heads, and a much larger           
 ┃   transformer dimension, scaling to 35M for the trainable config and 373M for the strong config    
 ┃   parameters (compared to the 280K parameter lite model). 
 tokens per state, you need 300,000 to 3,000,000 unique board        
 ┃   positions.                                                          
 ┃   • Completed Games: At an average of 150 to 200 plies per game, you  
 ┃   will need to generate between 2,000 and 15,000 teacher-guided MCTS  
 ┃   games to train this model to convergence.                           
 ┃   ──────                                                              
 ┃   ### 2. Recommended Training Strategy                                
 ┃                                                                       
 ┃   Given the MoE architecture, multiple auxiliary heads, and the larger
 ┃   parameter footprint, the current single-sample training script is   
 ┃   not viable. A successful strategy requires:                         
 ┃                                                                       
 ┃   #### A. Batched GPU Execution (MPS/CUDA)                            
 ┃                                                                       
 ┃   Unlike the lite model where CPU host-to-device kernel launch latency
 ┃   made CPU training faster, the larger model will easily saturate a   
 ┃   GPU. You must use batched training (batch size 64 to 256) with      
 ┃   padded sequence masking to feed the GPU's tensor cores efficiently. 
 ┃                                                                       
 ┃   #### B. Two-Phase Hybrid Training                                   
 ┃                                                                       
 ┃   1. Phase 1: Supervised Offline Distillation: Pre-train the model's  
 ┃   policy, value, and auxiliary heads on the 5,000+ static teacher-    
 ┃   generated dataset to establish solid playing heuristics.            
 ┃   2. Phase 2: AlphaZero Self-Play Loop: Once the supervised baseline  
 ┃   exceeds a 70% win rate against random play, switch to a self-play   
 ┃   reinforcement loop where the model plays against MCTS-enhanced      
 ┃   versions of itself to refine its tactical endgame.                  
 ┃                                                                       
 ┃   #### C. MoE Load-Balancing & Router Warmup                          
 ┃                                                                       
 ┃   MoE routers are prone to "representation collapse" early in training
 ┃   (where the router sends all tokens to a single "favorite" expert,   
 ┃   leaving others untrained). You must:                                
 ┃                                                                       
 ┃   • Train with the MoE load-balancing auxiliary loss enabled          
 ┃   (currently implemented in moe.py).                                  
 ┃   • Warm up the dense layers first for 5,000 steps before unfreezing  
 ┃   and training the router weights.                                    
 ┃                                                                       
 ┃   #### D. Dynamic Auxiliary Loss Weighting                            
 ┃                                                                       
 ┃   With 9 auxiliary heads (predicting piece availability, queen        
 ┃   surrounds, etc.) plus policy, value, and diffusion losses, the model
 ┃   is optimizing 12 separate gradients. You should use a dynamic       
 ┃   weighting scheme (such as GradNorm or strict scale factor schedules)
 ┃   to prevent the auxiliary tasks from overpowering the core policy    
 ┃   gradients.       