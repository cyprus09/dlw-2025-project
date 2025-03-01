import React from 'react';
import StructuredOutputDemo from '../components/StructuredOutputDemo';
import { Link as RouterLink } from 'react-router-dom';

const StructuredOutput: React.FC = () => {
  return (
    <div className="container mx-auto px-4">
      <div className="my-8">
        <h1 className="text-4xl font-bold mb-6">
          Structured Output
        </h1>
        <div className="mb-8">
          <RouterLink to="/" className="text-blue-600 hover:text-blue-800 transition-colors">
            Back to Home
          </RouterLink>
        </div>
        <StructuredOutputDemo />
      </div>
    </div>
  );
};

export default StructuredOutput;
