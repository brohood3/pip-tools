#!/bin/bash
# Script to run tool tests with different configurations
# This script runs test_tools.py which now tests multiple prompts per tool 
# and shows the first ~150 words of each response
# Test reports are automatically saved to the test_reports directory

# Make test_tools.py executable
chmod +x test_tools.py

# Create reports directory if it doesn't exist
mkdir -p test_reports

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Trading Tools API Testing Script${NC}"
echo "=============================="
echo -e "${BLUE}This script tests each tool with multiple prompts and displays response previews${NC}"
echo -e "${BLUE}Test reports are automatically saved to the 'test_reports' directory${NC}"

# Function to display the menu
show_menu() {
    echo
    echo "Choose a testing option:"
    echo "1. Test all tools with default model (gemini-2.0-flash)"
    echo "2. Test all tools with multiple models (default, gpt-4o, claude-3-haiku)"
    echo "3. Test specific tool with default model"
    echo "4. Test specific tool with specific model"
    echo "5. View recent test reports"
    echo "6. Exit"
    echo
    echo -n "Enter your choice (1-6): "
}

# Function to get tool selection
select_tool() {
    echo
    echo "Available tools:"
    echo "1. ten_word_ta - Ten-word technical analysis"
    echo "2. general_predictor - General market prediction"
    echo "3. tool_selector - Tool selection based on user intent"
    echo "4. technical_analysis - Comprehensive technical analysis"
    echo "5. macro_outlook_analyzer - Macro economic outlook analysis"
    echo "6. fundamental_analysis - Fundamental token analysis"
    echo "7. lunar_crush_screener - Screening for promising cryptocurrencies"
    echo "8. query_extract - Extracting information from Dune queries"
    echo "9. price_predictor - Detailed price prediction with scenarios"
    echo
    echo -n "Enter tool number (1-9): "
    read tool_num
    
    case $tool_num in
        1) selected_tool="ten_word_ta" ;;
        2) selected_tool="general_predictor" ;;
        3) selected_tool="tool_selector" ;;
        4) selected_tool="technical_analysis" ;;
        5) selected_tool="macro_outlook_analyzer" ;;
        6) selected_tool="fundamental_analysis" ;;
        7) selected_tool="lunar_crush_screener" ;;
        8) selected_tool="query_extract" ;;
        9) selected_tool="price_predictor" ;;
        *) echo "Invalid selection"; selected_tool="" ;;
    esac
    
    echo
}

# Function to get model selection
select_model() {
    echo
    echo "Available models:"
    echo "1. gpt-4o - OpenAI's GPT-4o"
    echo "2. claude-3-haiku - Anthropic's Claude 3 Haiku"
    echo "3. gemini-pro - Google's Gemini Pro"
    echo "4. gemini-2.0-flash - Google's Gemini 2.0 Flash (default)"
    echo "5. mistral-medium - Mistral Medium"
    echo "6. llama-3-8b - Meta's Llama 3 8B"
    echo
    echo -n "Enter model number (1-6): "
    read model_num
    
    case $model_num in
        1) selected_model="gpt-4o" ;;
        2) selected_model="claude-3-haiku" ;;
        3) selected_model="gemini-pro" ;;
        4) selected_model="gemini-2.0-flash" ;;
        5) selected_model="mistral-medium" ;;
        6) selected_model="llama-3-8b" ;;
        *) echo "Invalid selection"; selected_model="" ;;
    esac
    
    echo
}

# Function to view recent test reports
view_reports() {
    if [ ! -d "test_reports" ] || [ -z "$(ls -A test_reports)" ]; then
        echo -e "${YELLOW}No test reports found in the test_reports directory.${NC}"
        return
    fi
    
    echo -e "${BLUE}Recent test reports:${NC}"
    echo
    
    # List the 5 most recent text reports
    reports=($(ls -t test_reports/*.txt | head -5))
    
    if [ ${#reports[@]} -eq 0 ]; then
        echo "No text reports found."
        return
    fi
    
    for i in "${!reports[@]}"; do
        report="${reports[$i]}"
        filename=$(basename "$report")
        date=$(date -r "$report" "+%Y-%m-%d %H:%M:%S")
        
        echo "$((i+1)). $filename (created on $date)"
    done
    
    echo
    echo -n "Enter a number to view a report (or 0 to cancel): "
    read report_num
    
    if [ "$report_num" -gt 0 ] && [ "$report_num" -le "${#reports[@]}" ]; then
        selected_report="${reports[$((report_num-1)))]}"
        echo
        echo -e "${GREEN}Report: $(basename "$selected_report")${NC}"
        echo -e "${YELLOW}$(cat "$selected_report")${NC}"
    elif [ "$report_num" -ne 0 ]; then
        echo -e "${YELLOW}Invalid selection.${NC}"
    fi
}

# Main loop
while true; do
    show_menu
    read choice
    
    case $choice in
        1)
            echo -e "${YELLOW}Testing all tools with default model...${NC}"
            ./test_tools.py
            ;;
        2)
            echo -e "${YELLOW}Testing all tools with multiple models...${NC}"
            ./test_tools.py --models gpt-4o claude-3-haiku
            ;;
        3)
            select_tool
            if [ ! -z "$selected_tool" ]; then
                echo -e "${YELLOW}Testing $selected_tool with default model...${NC}"
                ./test_tools.py --tools $selected_tool
            fi
            ;;
        4)
            select_tool
            if [ ! -z "$selected_tool" ]; then
                select_model
                if [ ! -z "$selected_model" ]; then
                    echo -e "${YELLOW}Testing $selected_tool with $selected_model...${NC}"
                    ./test_tools.py --tools $selected_tool --models $selected_model
                fi
            fi
            ;;
        5)
            view_reports
            ;;
        6)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid option. Please try again."
            ;;
    esac
done 