#!/usr/bin/env python3

import pygame
import numpy as np
import sys

def create_sample_rgb_array(width=800, height=600):
    """Create a sample RGB numpy array with a gradient pattern"""
    # Create coordinate grids
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Create RGB channels
    red = (X * 255).astype(np.uint8)
    green = (Y * 255).astype(np.uint8) 
    blue = ((X + Y) * 127.5).astype(np.uint8)
    
    # Stack into RGB array
    rgb_array = np.stack([red, green, blue], axis=2)
    return rgb_array

def display_rgb_array(rgb_array):
    """Display RGB numpy array using pygame"""
    pygame.init()
    
    height, width = rgb_array.shape[:2]
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("RGB Array Display")
    
    # Convert numpy array to pygame surface
    surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Display the surface
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

def main():
    # Create a sample RGB array
    rgb_array = create_sample_rgb_array()
    
    print(f"Displaying RGB array of shape: {rgb_array.shape}")
    print("Press ESC or close window to exit")
    
    # Display it
    display_rgb_array(rgb_array)

if __name__ == "__main__":
    main() 