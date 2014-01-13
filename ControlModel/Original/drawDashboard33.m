global problem;
global currentTask;
global lastTask;
global dashboard;


set(0,'CurrentFigure',dashboard);


clf;
hold on;



subplot(2,3,1);
bar(problem.belief);
title('Belief State')
set(gca,'XTickLabel',logger.frequenciesUsed)
ylim([0 1.2]);

subplot(2,3,4);
bar(problem.eRewards(:,1));
set(gca,'XTickLabel', problem.controls);
ylim([-40 20]);
subplot(2,3,5);
bar(problem.eRewards(:,2));
set(gca,'XTickLabel', problem.controls);
ylim([-40 20]);
subplot(2,3,6);
bar(problem.eRewards(:,3));
set(gca,'XTickLabel', problem.controls);
ylim([-40 20]);

subplot(2,3,3);
set(gca,'YTickLabel',{});
set(gca,'XTickLabel',{});

text(.05,.9,strcat('Last action: ',{' '},problem.actionLabels(problem.lastActionIndex)),...
    'FontSize',14,'HorizontalAlignment','left')
text(.05,.82,strcat('Last classification: ',{' '},num2str(problem.lastClass)),...
    'FontSize',14,'HorizontalAlignment','left')
text(.05,.74,strcat('Last task: ',{' '},lastTask),...
    'FontSize',14,'HorizontalAlignment','left')
text(.05,.58,strcat('Last control: ',{' '},problem.controlMade),...
    'FontSize',14,'HorizontalAlignment','left')
text(.05,.50,strcat('Last feedback: ',{' '},num2str(problem.feedback)),...
    'FontSize',14,'HorizontalAlignment','left')
text(.05,.42,strcat('P(Ct|Ot): ',{' '},num2str(problem.observation)),...
    'FontSize',14,'HorizontalAlignment','left')
text(.05,.34,strcat('Current task: ',{' '},num2str(currentTask)),...
    'FontSize',14,'HorizontalAlignment','left')

set(gca,'Color',[0.8 0.8 0.8]);
set(gca,'Box','On');
set(gca,'TickLength',[0 0]);
title('Agent Status')
hold off;