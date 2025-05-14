import React from 'react';
import './ResultDisplay.css';

const ResultDisplay = ({ result }) => {
    if (!result || !result.length) {
        return null;
    }

    // The backend now sends predictions directly as an array
    const allPredictions = result;

    // No need for emoji mapping as backend now provides emojis

    const getConfidenceColor = (confidence) => {
        const percent = parseFloat(confidence.replace('%', ''));
        if (percent > 90) return '#28a745';
        if (percent > 70) return '#ffc107';
        return '#dc3545';
    };

    const getEncouragingMessage = (confidence) => {
        const percent = parseFloat(confidence.replace('%', ''));
        if (percent > 90) return "I'm super sure about this! 🌟";
        if (percent > 70) return "I think I got it right! 😊";
        return "I'm not completely sure, but let's try! 🤔";
    };

    // Use the predictions directly from the backend
    const predictions = allPredictions;

    return (
        <div className="result-container">
            <h2 className="result-header">Here's what I found! 🎯</h2>
            
            <div className="main-prediction">
                <div className="prediction-emoji">{predictions[0].emoji}</div>
                <h3>This might be a {predictions[0].class}...</h3>
                <div className="confidence-bar">
                    <div 
                        className="confidence-fill"
                        style={{ 
                            width: predictions[0].confidence,
                            backgroundColor: getConfidenceColor(predictions[0].confidence)
                        }}
                    ></div>
                </div>
                <p className="confidence-text">
                    {getEncouragingMessage(predictions[0].confidence)}
                </p>
            </div>

            <div className="object-definition">
                <h4>Fun Facts About {predictions[0].class}s:</h4>
                <div className="definition-content">
                    {/* Extract the fun facts from the message */}
                    {predictions[0].message && (
                        <>
                            {/* Split the message by the phrase "Did you know?" or "Fun fact:" or "Interesting fact:" */}
                            {(() => {
                                const message = predictions[0].message;
                                let facts = [];
                                
                                // Check if the message contains any of our fact markers
                                if (message.includes("Did you know?")) {
                                    const factPart = message.split("Did you know?")[1].trim();
                                    facts.push(factPart);
                                } else if (message.includes("Fun fact:")) {
                                    const factPart = message.split("Fun fact:")[1].trim();
                                    facts.push(factPart);
                                } else if (message.includes("Interesting fact:")) {
                                    const factPart = message.split("Interesting fact:")[1].trim();
                                    facts.push(factPart);
                                }
                                
                                // If we found facts, display them
                                if (facts.length > 0) {
                                    // Split facts if they contain multiple sentences
                                    const sentences = facts[0].split(/\.|!/).filter(s => s.trim().length > 0);
                                    
                                    return sentences.map((sentence, index) => (
                                        <p key={index}><span className="fact-bullet">🔹</span> {sentence.trim()}</p>
                                    ));
                                } else {
                                    // If no specific facts found, display the whole message
                                    return (
                                        <p><span className="fact-bullet">🔹</span> {message}</p>
                                    );
                                }
                            })()}
                        </>
                    )}
                    
                    {/* If no message is available, show default facts */}
                    {!predictions[0].message && (
                        <>
                            <p><span className="fact-bullet">🔹</span> Learning about different objects helps us understand the world around us!</p>
                            <p><span className="fact-bullet">🔹</span> Scientists study objects to learn how they work and what they're made of.</p>
                            <p><span className="fact-bullet">🔹</span> Can you think of other interesting {predictions[0].class}s you've seen before?</p>
                        </>
                    )}
                </div>
            </div>

            <div className="fun-fact">
                <p>Did you know? 🤓</p>
                <p>I can recognize 50 different objects! Try showing me something else!</p>
            </div>

            <button 
                className="try-again-button"
                onClick={() => window.location.reload()}
            >
                Let's Try Another! 🎮
            </button>
        </div>
    );
};

export default ResultDisplay;
