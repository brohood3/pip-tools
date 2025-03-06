document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const connectWalletBtn = document.getElementById('connectWalletBtn');
    const signInBtn = document.getElementById('signInBtn');
    const walletInfo = document.getElementById('walletInfo');
    const walletAddressDisplay = document.getElementById('walletAddress');
    const errorMessage = document.getElementById('errorMessage');
    
    // State variables
    let currentAccount = null;
    let walletProvider = null;
    let nonce = null;
    let chainId = null;
    
    // Initialize
    checkSession();
    
    // Event Listeners
    connectWalletBtn.addEventListener('click', connectWallet);
    signInBtn.addEventListener('click', signInWithEthereum);
    
    // Functions
    async function checkSession() {
        try {
            const response = await fetch('/auth/session');
            const data = await response.json();
            
            if (data.authenticated) {
                // User is already authenticated
                window.location.href = '/';
            }
        } catch (error) {
            console.error('Failed to check session:', error);
        }
    }
    
    async function connectWallet() {
        try {
            // Show loading state
            connectWalletBtn.disabled = true;
            connectWalletBtn.innerHTML = '<span class="loading"></span> Connecting...';
            
            // Clear previous errors
            hideError();
            
            // Check if MetaMask or other wallet is available
            if (window.ethereum) {
                try {
                    // Request account access
                    const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
                    walletProvider = new ethers.providers.Web3Provider(window.ethereum);
                    
                    // Get connected chain
                    const network = await walletProvider.getNetwork();
                    chainId = network.chainId;
                    
                    // Handle account selection
                    handleAccountsChanged(accounts);
                    
                    // Listen for account changes
                    window.ethereum.on('accountsChanged', handleAccountsChanged);
                    
                    // Get nonce from server
                    await getNonce(currentAccount);
                    
                    // Reset connect button
                    connectWalletBtn.disabled = false;
                    connectWalletBtn.innerHTML = '<i class="fa-solid fa-wallet"></i> Connected';
                    
                    // Show sign-in button
                    signInBtn.style.display = 'flex';
                    
                } catch (error) {
                    console.error(error);
                    showError('Failed to connect wallet: ' + getErrorMessage(error));
                    resetConnectButton();
                }
            } else {
                showError('No Ethereum wallet detected. Please install MetaMask or another wallet provider.');
                resetConnectButton();
            }
        } catch (error) {
            console.error(error);
            showError('An error occurred: ' + getErrorMessage(error));
            resetConnectButton();
        }
    }
    
    function handleAccountsChanged(accounts) {
        if (accounts.length === 0) {
            // User has disconnected all accounts
            showError('Please connect to a wallet.');
            resetConnection();
        } else {
            // Update current account
            currentAccount = accounts[0];
            
            // Display wallet info
            walletAddressDisplay.textContent = currentAccount;
            walletInfo.style.display = 'block';
        }
    }
    
    async function getNonce(address) {
        try {
            const response = await fetch(`/auth/nonce?address=${address}`);
            const data = await response.json();
            nonce = data.nonce;
        } catch (error) {
            console.error('Failed to get nonce:', error);
            showError('Failed to get authentication challenge from server.');
            throw error;
        }
    }
    
    async function signInWithEthereum() {
        try {
            // Show loading state
            signInBtn.disabled = true;
            signInBtn.innerHTML = '<span class="loading"></span> Signing...';
            
            // Clear previous errors
            hideError();
            
            if (!currentAccount || !nonce || !chainId) {
                showError('Wallet connection issue. Please reconnect your wallet.');
                resetSignInButton();
                return;
            }
            
            // Prepare the message
            const domain = window.location.host;
            const origin = window.location.origin;
            const statement = 'Sign in with Ethereum to Eolas AI Chat.';
            const issuedAt = new Date().toISOString();
            
            // Create SIWE message string in the correct format
            const message = `${domain} wants you to sign in with your Ethereum account:
${currentAccount}

${statement}

URI: ${origin}
Version: 1
Chain ID: ${chainId}
Nonce: ${nonce}
Issued At: ${issuedAt}`;

            console.log("Message to be signed:", message);
            
            // Get the signer from the provider
            const signer = walletProvider.getSigner();
            
            try {
                // Request personal signature from user for better compatibility
                const signature = await window.ethereum.request({
                    method: 'personal_sign',
                    params: [message, currentAccount]
                });
                
                console.log("Signature:", signature);
                
                // Send to server for verification
                await verifySignature(message, signature);
                
            } catch (error) {
                console.error('Signature error:', error);
                
                if (error.code === 4001) {
                    // User rejected the signature request
                    showError('You need to sign the message to authenticate.');
                } else {
                    showError('Failed to sign message: ' + getErrorMessage(error));
                }
                
                resetSignInButton();
            }
        } catch (error) {
            console.error('Sign-in error:', error);
            showError('Authentication error: ' + getErrorMessage(error));
            resetSignInButton();
        }
    }
    
    async function verifySignature(message, signature) {
        try {
            const response = await fetch('/auth/verify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message,
                    signature
                }),
            });
            
            const data = await response.json();
            
            if (response.ok && data.success) {
                // Redirect to the main page after successful authentication
                window.location.href = '/';
            } else {
                // Show error from server
                showError(data.error || 'Authentication failed.');
                resetSignInButton();
            }
        } catch (error) {
            console.error('Verification error:', error);
            showError('Failed to verify your signature with the server.');
            resetSignInButton();
        }
    }
    
    // Helper functions
    function resetConnection() {
        currentAccount = null;
        nonce = null;
        chainId = null;
        walletInfo.style.display = 'none';
        signInBtn.style.display = 'none';
        resetConnectButton();
    }
    
    function resetConnectButton() {
        connectWalletBtn.disabled = false;
        connectWalletBtn.innerHTML = '<i class="fa-solid fa-wallet"></i> Connect Wallet';
    }
    
    function resetSignInButton() {
        signInBtn.disabled = false;
        signInBtn.innerHTML = '<i class="fa-solid fa-signature"></i> Sign Message to Login';
    }
    
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
    }
    
    function hideError() {
        errorMessage.textContent = '';
        errorMessage.style.display = 'none';
    }
    
    function getErrorMessage(error) {
        return error.message || error.reason || error.toString();
    }
}); 