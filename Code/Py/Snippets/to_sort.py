

"""

[&&&&&&&&&&&&&&&&&&] | [&&&&&&&&&&&&&&&&&&&&]


"""

def print_inplace(step, err, acc): #.format('STEP', 'ERROR', 'ACCURACY')
    #time.sleep(0.1)
    title = '{:<5}:  {:^7}   |  {:^7}'
    body  = '{:>5}:  {:.5f}   |   {:.4f}'
    s, e, a = step+1, float(err), float(acc)
    S, E, A = 'STEP', 'ERROR', 'ACCURACY'
    O = '\r{}'.format(body)
    ##status = O.format(S,E,A,s,e,a)
    if step == 0:
        print('\n' + title.format(S,E,A))
    sys.stdout.write(O.format(s,e,a))
    sys.stdout.flush()
    #time.sleep(0.05)
    #l = 40
    #e = round(l * float(err))
    #a = round(l * float(acc))
    #bar = '\r[{: <40}] | [{: <40}]'.format('@'*e, '@'*a)
    #sys.stdout.write(bar)
    #sys.stdout.flush()
