import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import ChatBot from '../src/ChatBox';

describe('ChatBox Component', () => {
  it('renders without crashing', () => {
    render(<ChatBot chatHistory={[]} />);
    expect(screen.getByRole('button', { name: /Open AI Doctor chat/i })).toBeInTheDocument();
  });
});
