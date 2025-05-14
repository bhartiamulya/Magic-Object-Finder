import React, { useState } from 'react';
import ImageUpload from './ImageUpload';
import ResultDisplay from './ResultDisplay';
import './App.css';

function App() {
  const [predictions, setPredictions] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const categories = [
    'apple 🍎', 'aquarium fish 🐠', 'baby 👶', 'bear 🐻', 'beaver 🦫',
    'bed 🛏️', 'bee 🐝', 'beetle 🪲', 'bicycle 🚲', 'bottle 🍾',
    'bowl 🥣', 'boy 👦', 'bridge 🌉', 'bus 🚌', 'butterfly 🦋',
    'camel 🐪', 'can 🥫', 'castle 🏰', 'caterpillar 🐛', 'cattle 🐄',
    'chair 🪑', 'chimpanzee 🦧', 'clock ⏰', 'cloud ☁️', 'cockroach 🪳',
    'couch 🛋️', 'crab 🦀', 'crocodile 🐊', 'cup ☕', 'dinosaur 🦖',
    'dolphin 🐬', 'elephant 🐘', 'flatfish 🐟', 'flower 🌸', 'forest 🌳',
    'fox 🦊', 'girl 👧', 'hamster 🐹', 'house 🏠', 'kangaroo 🦘',
    'keyboard ⌨️', 'lamp 💡', 'lawn mower 🚜', 'leopard 🐆', 'lion 🦁',
    'lizard 🦎', 'lobster 🦞', 'man 👨', 'maple tree 🍁', 'mountain ⛰️'
  ];

  return (
    <div className="App">
      <div className="snowflake1"></div>
      <div className="snowflake2"></div>
      <div className="snowflake3"></div>
      <div className="snowflake4"></div>
      <div className="snowflake5"></div>
      <div className="snowflake6"></div>
      <div className="snowflake7"></div>
      <div className="snowflake8"></div>
      <div className="snowflake9"></div>
      <div className="snowflake10"></div>
      <div className="snowflake11"></div>
      <div className="snowflake12"></div>
      <div className="snowflake13"></div>
      <div className="snowflake14"></div>
      <div className="snowflake15"></div>
      <div className="snowflake16"></div>
      <div className="snowflake17"></div>
      <div className="snowflake18"></div>
      <div className="snowflake19"></div>
      <div className="snowflake20"></div>
      <div className="mickey-top-right"></div>
      <div className="mickey-bottom-left"></div>
      <div className="candy-top-left"></div>
      <div className="candy-bottom-right"></div>
      <header className="App-header">
        <h1>🔍 Magic Object Finder</h1>
        <p className="subtitle">I can recognize 50 different things! Let's play and learn! 🎮</p>
      </header>

      <ImageUpload 
        setError={setError}
        setIsLoading={setIsLoading}
        setPredictions={setPredictions}
      />

      {error && (
        <div className="error-message">
          <p>Oops! 😅 {error}</p>
          <button onClick={() => setError(null)}>Try Again! 🔄</button>
        </div>
      )}

      {isLoading && (
        <div className="loading-message">
          <p>Looking very carefully... 🔍</p>
          <div className="loading-spinner">🤔</div>
        </div>
      )}

      {predictions && !isLoading && (
        <ResultDisplay result={predictions} />
      )}

      <footer className="App-footer">
        <p>Made with ❤️ for curious minds!</p>
      </footer>
      <div className="snow-pile"></div>
    </div>
  );
}

export default App;
