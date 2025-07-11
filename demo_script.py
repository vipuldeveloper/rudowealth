#!/usr/bin/env python3
"""
RudoWealth RAG Demo Script for AI Interview
Demonstrates the Smart Portfolio Allocation Engine
"""

import requests
import json
import time
from typing import Dict, Any

class RudoWealthDemo:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def health_check(self) -> Dict[str, Any]:
        """Check if the server is running"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_rag_process(self, query: str) -> Dict[str, Any]:
        """Get RAG process breakdown for a query"""
        try:
            response = requests.get(f"{self.base_url}/demo/rag-process", params={"query": query})
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def chat_query(self, message: str) -> Dict[str, Any]:
        """Send a chat query and get response"""
        try:
            response = requests.post(f"{self.base_url}/chat", json={"message": message})
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_user_profiles(self) -> Dict[str, Any]:
        """Get all user profiles"""
        try:
            response = requests.get(f"{self.base_url}/demo/user-profiles")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_strategies(self) -> Dict[str, Any]:
        """Get all investment strategies"""
        try:
            response = requests.get(f"{self.base_url}/demo/strategies")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_market_regime(self) -> Dict[str, Any]:
        """Get current market regime"""
        try:
            response = requests.get(f"{self.base_url}/demo/market-regime")
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def print_section(title: str, content: Any):
    """Print a formatted section"""
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print(f"{'='*60}")
    if isinstance(content, dict):
        print(json.dumps(content, indent=2))
    else:
        print(content)

def demo_flow():
    """Main demo flow for the interview"""
    demo = RudoWealthDemo()
    
    print("🚀 RudoWealth RAG Demo - Smart Portfolio Allocation Engine")
    print("=" * 80)
    
    # Step 1: Setup (30 seconds)
    print_section("STEP 1: SYSTEM SETUP", "Checking RAG-powered investment advisor...")
    
    health = demo.health_check()
    if "error" in health:
        print(f"❌ Server not running: {health['error']}")
        print("Please start the server with: python main.py")
        return
    
    print_section("SYSTEM STATUS", health)
    
    # Step 2: Query Demonstration (2 minutes)
    print_section("STEP 2: PERSONALIZED QUERY DEMONSTRATION", 
                  "Query: 'RUDO001 wants to invest ₹5 lakhs bonus'")
    
    # Show RAG process breakdown
    rag_breakdown = demo.get_rag_process("RUDO001 wants to invest ₹5 lakhs bonus")
    print_section("RAG PROCESS BREAKDOWN", rag_breakdown)
    
    # Get actual response
    response = demo.chat_query("RUDO001 wants to invest ₹5 lakhs bonus")
    print_section("AI RESPONSE", response.get("response", "No response"))
    
    # Step 3: Real-time Intelligence (2 minutes)
    print_section("STEP 3: REAL-TIME MARKET REGIME INTELLIGENCE", 
                  "Query: 'What if market enters recession tomorrow?'")
    
    # Show recession scenario
    market_data = demo.get_market_regime()
    print_section("CURRENT MARKET REGIME", market_data.get("market_regime", {}).get("current_regime", {}))
    
    # Get recession response
    recession_response = demo.chat_query("What if market enters recession tomorrow?")
    print_section("RECESSION STRATEGY RESPONSE", recession_response.get("response", "No response"))
    
    # Step 4: Behavioral AI (2 minutes)
    print_section("STEP 4: BEHAVIORAL AI INTERVENTION", 
                  "Query: 'RUDO001 wants to exit all equity investments today'")
    
    # Show user profile
    profiles = demo.get_user_profiles()
    rudo001_profile = profiles.get("user_profiles", {}).get("RUDO001", {})
    print_section("RUDO001 BEHAVIORAL PROFILE", rudo001_profile)
    
    # Get behavioral intervention response
    behavioral_response = demo.chat_query("RUDO001 wants to exit all equity investments today")
    print_section("BEHAVIORAL INTERVENTION RESPONSE", behavioral_response.get("response", "No response"))
    
    # Step 5: Technical Architecture Overview
    print_section("STEP 5: TECHNICAL ARCHITECTURE", 
                  "RAG Components and Data Flow")
    
    strategies = demo.get_strategies()
    print_section("INVESTMENT STRATEGIES LOADED", f"Total: {len(strategies.get('strategies', {}))}")
    
    print_section("RAG ARCHITECTURE SUMMARY", """
    User Query → Vector Search (FAISS) → Context Retrieval → LLM Generation → Response
         ↓              ↓                      ↓               ↓
       Parsing    [User Profile +         Prompt Building   Personalized
                  Market Data +                             Financial
                  Strategy Rules]                           Advice
    """)
    
    print_section("DEMO COMPLETED", """
    ✅ Multi-modal RAG (structured + unstructured data)
    ✅ Real-time market regime integration
    ✅ Personalized user profiles
    ✅ Behavioral intervention strategies
    ✅ Scalable architecture for 10,000+ users
    """)

if __name__ == "__main__":
    demo_flow() 