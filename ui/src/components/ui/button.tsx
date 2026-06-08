/**
 * Button — shadcn-style component (cva + Slot) styled with the Orbiter
 * palette baked into each variant. Replaces the ad-hoc `cls.btn`, `cls.btnGo`,
 * `cls.btnDanger`, `cls.btnSuccess`, `SMALL_BTN`, `ARROW_BTN` patterns we had
 * scattered across the UI.
 *
 * Use `asChild` to render as a different element (e.g. wrap an `<a>` in the
 * button's styling, like the Library "Download all" link).
 */

import { forwardRef, type ButtonHTMLAttributes } from 'react';
import { Slot } from '@radix-ui/react-slot';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '../../lib/utils';

const buttonVariants = cva(
  'inline-flex items-center justify-center rounded-md border font-medium ' +
    'tracking-tight transition-colors ' +
    'focus-visible:outline-1 focus-visible:outline-accent ' +
    'disabled:pointer-events-none disabled:opacity-40',
  {
    variants: {
      variant: {
        // Primary action — soft indigo, no heavy fill.
        default:
          'bg-indigo-500/85 border-indigo-400/40 text-white ' +
          'hover:bg-indigo-500 hover:border-indigo-400/70',
        secondary:
          'bg-zinc-900 border-zinc-800 text-zinc-200 ' +
          'hover:bg-zinc-800 hover:border-zinc-700',
        destructive:
          'bg-red-950/70 border-red-900/80 text-red-200 ' +
          'hover:bg-red-900/70 hover:border-red-800',
        success:
          'bg-emerald-950/70 border-emerald-900/80 text-emerald-200 ' +
          'hover:bg-emerald-900/70 hover:border-emerald-800',
        ghost:
          'border-transparent bg-transparent text-zinc-400 ' +
          'hover:bg-zinc-900 hover:text-zinc-100',
        outline:
          'border-zinc-800 bg-transparent text-zinc-400 ' +
          'hover:border-zinc-700 hover:bg-zinc-900 hover:text-zinc-100',
        tab:
          'border-transparent bg-transparent rounded-md text-zinc-500 ' +
          'hover:text-zinc-200 data-[state=active]:bg-zinc-900 ' +
          'data-[state=active]:text-zinc-50 data-[state=active]:border-zinc-800',
      },
      size: {
        default: 'h-8 px-3 text-[13px]',
        sm: 'h-7 px-2 text-[12px]',
        xs: 'h-6 px-2 text-[11px]',
        lg: 'h-9 px-4 text-[14px]',
        icon: 'h-8 w-8 text-[14px] p-0',
      },
    },
    defaultVariants: { variant: 'default', size: 'default' },
  },
);

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : 'button';
    return (
      <Comp
        ref={ref}
        className={cn(buttonVariants({ variant, size }), className)}
        {...props}
      />
    );
  },
);
Button.displayName = 'Button';

export { buttonVariants };
