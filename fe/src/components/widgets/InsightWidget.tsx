import React from 'react';
import { cn } from '@/lib/utils';

interface InsightWidgetProps extends React.HTMLAttributes<HTMLDivElement> {
    title: string;
    children: React.ReactNode;
}

const InsightWidget = React.forwardRef<HTMLDivElement, InsightWidgetProps>(
    ({ title, children, className, ...props }, ref) => {
        return (
            <div
                ref={ref}
                className={cn(
                    'rounded-lg border bg-card/50 text-card-foreground shadow-sm',
                    'p-4 h-full flex flex-col',
                    className
                )}
                {...props}
            >
                <h3 className='font-semibold leading-none tracking-tight mb-1 text-xl'>
                    {title}
                </h3>
                <hr className='border-t border-gray-400 my-2' />
                <div className='flex-1 text-sm'>{children}</div>
            </div>
        );
    }
);

InsightWidget.displayName = 'InsightWidget';

export default InsightWidget;
