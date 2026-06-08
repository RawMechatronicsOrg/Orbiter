/** Input — shadcn-style wrapper around the project's existing field styling. */

import { forwardRef, type InputHTMLAttributes } from 'react';
import { cn } from '../../lib/utils';

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  ({ className, type = 'text', ...props }, ref) => (
    <input
      ref={ref}
      type={type}
      className={cn(
        'w-full rounded-md border border-fieldline bg-field px-2.5 py-1',
        'font-mono text-[13px] text-zinc-100',
        'focus:outline-1 focus:outline-accent focus:border-accent/40',
        'disabled:cursor-not-allowed disabled:opacity-40',
        className,
      )}
      {...props}
    />
  ),
);
Input.displayName = 'Input';
